from CUB.dataset import load_data
import torch
import argparse
from tqdm import tqdm
from pathlib import Path


class ModelGradWrap:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        weight_decay: float | None = None,
    ) -> None:
        super().__init__()
        if weight_decay is not None:
            self._weight_decay_loss = torch.sum(torch.cat([
                param.square().view(-1) for param in self.params
            ])) * weight_decay
        else:
            self._weight_decay_loss = 0
        
        self._loss_fn = loss_fn
        self._model = model.eval()
        names, params = list(), list()
        for n, p in self._model.named_parameters():
            names.append(n)
            params.append(p)
        tmp = set(names)
        assert tmp == (tmp & model.state_dict().keys())
        self._names = tuple(names)
        self._params = tuple(params)

    @property
    def params(self): return self._params

    def load_params(self, params: tuple[torch.Tensor]):
        self._params = params
        old = self._model.state_dict()
        old.update(dict(zip(self._names, self._params)))
        self._model.load_state_dict(old)

    def forward(self, inputs):
        return self._model(inputs)

    def grad(
        self,
        gt: torch.Tensor,
        inputs: torch.Tensor,
        time=1,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        out = self.forward(inputs)
        if isinstance(out, list): out = torch.cat(out, -1).float()
        masked = gt.isnan()
        gt[masked] = out[masked]
        assert not gt.isnan().any()
        loss = self._loss_fn(gt, out)

        grad = [loss + self._weight_decay_loss]
        for i in range(1, time+1):
            grad = torch.autograd.grad(
                outputs=map(lambda x: x.sum(), grad),
                inputs=self.params,
                create_graph=(i < time),
                allow_unused=True,
            )
            grad = [x if x is not None else torch.zeros_like(self.params[i], requires_grad=True) for i, x in enumerate(grad)]
        return tuple(grad)


def compute_s_test(
    model_grad: ModelGradWrap,
    data_loader: torch.utils.data.DataLoader,
    sample_data: None | dict[str, torch.Tensor] = None,
    epochs: int = 1,
) -> list[torch.FloatTensor]:
    r'''
    Compute the s_test vector, which is an approximation of the Hessian inverse times the gradients
    $s_{test}=H_{\hat{\theta}}^{-1}\partial_{theta}L(z_{test},\hat{\theta})$
    '''
    if sample_data is not None:
        init = model_grad.grad(time=1, **sample_data)
    else:
        loader_iter = iter(data_loader)
        init = model_grad.grad(time=1, **next(loader_iter))
        for data in loader_iter:
            grad = model_grad.grad(time=1, **next(loader_iter))
            with torch.no_grad():
                for x, y in zip(init, grad):
                    x += y
                    #assert not x.isnan().any()

    result = tuple([x.detach().clone() for x in init])

    for epoch in range(1, 1+epochs):
        for data in tqdm(data_loader, desc=f'compute_s_test {epoch}/{epochs}'):
            grad2 = model_grad.grad(time=2, **data)
            with torch.no_grad():
                for h, v, d in zip(result, init, grad2):
                    # NOTE: Inplace to avoid copy
                    h -= d * h
                    h += v
                    #assert not h.isnan().any()
    return result


def compute_single_level_influence(
    model_x2c: torch.nn.Module,
    model_c2y: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    sample_data: dict[str, torch.Tensor],
    weight_decay: float | None = None,
) -> tuple[torch.Tensor]:
    class _LoaderG:
        @torch.no_grad
        def __iter__(self):
            for data in data_loader:
                data['gt'] = data.pop('concept')
                yield data
    class _LoaderF:
        @torch.no_grad
        def __iter__(self):
            for data in data_loader:
                x = data['inputs']
                concept_inputs = model_x2c.forward(x)
                concept_inputs = torch.cat(concept_inputs, -1)
                #if self._model_g is not None:
                #    concept_inputs = concept_inputs - self._model_g.forward(x)
                yield {
                    'inputs': concept_inputs,
                    'gt': data['gt'],
                }

    model_g = ModelGradWrap(model_x2c, loss_fn, weight_decay=weight_decay)
    model_f = ModelGradWrap(model_c2y, loss_fn, weight_decay=weight_decay)
    h_inv_partial_f = compute_s_test(model_f, _LoaderF())

    sample_data['gt'] = sample_data.pop('concept')
    h_inv_partial_g_con = compute_s_test(model_g, _LoaderG(), sample_data)
    sample_data['gt'] = torch.zeros_like(sample_data['gt'])
    h_inv_partial_g_zero = compute_s_test(model_g, _LoaderG(), sample_data)

    new_param = [(x-(y-z)) for x,y,z in zip(model_g.params, h_inv_partial_g_zero, h_inv_partial_g_con)]
    model_g.load_params(new_param)
    h_inv_partial_complex = compute_s_test(model_f, _LoaderF())

    result = tuple([x+(y-z) for x,y,z in zip(model_f.params, h_inv_partial_f, h_inv_partial_complex)])
    return result


def compute_single_level_influence_v2(
    model_x2c: torch.nn.Module,
    model_c2y: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    sample_data: dict[str, torch.Tensor],
    weight_decay: float | None = None,
) -> tuple[torch.Tensor]:
    class _LoaderG:
        @torch.no_grad
        def __iter__(self):
            for data in data_loader:
                data['gt'] = data.pop('concept')
                yield data
    class _LoaderF:
        @torch.no_grad
        def __iter__(self):
            for data in data_loader:
                x = data['inputs']
                concept_inputs = model_x2c.forward(x)
                concept_inputs = torch.cat(concept_inputs, -1)
                #if self._model_g is not None:
                #    concept_inputs = concept_inputs - self._model_g.forward(x)
                yield {
                    'inputs': concept_inputs,
                    'gt': data['gt'],
                }

    model_g = ModelGradWrap(model_x2c, loss_fn, weight_decay=weight_decay)
    model_f = ModelGradWrap(model_c2y, loss_fn, weight_decay=weight_decay)

    h_inv_partial_f = None
    for data in _LoaderF():
        delta = compute_s_test(model_f, _LoaderF(), data)
        if h_inv_partial_f is None: h_inv_partial_f = delta
        h_inv_partial_f = tuple([x+y for x,y in zip(h_inv_partial_f, delta)])

    sample_data['gt'] = sample_data.pop('concept')
    h_inv_partial_g_con = compute_s_test(model_g, _LoaderG(), sample_data)
    sample_data['gt'] = torch.zeros_like(sample_data['gt'])
    h_inv_partial_g_zero = compute_s_test(model_g, _LoaderG(), sample_data)

    new_param = [(x-(y-z)) for x,y,z in zip(model_g.params, h_inv_partial_g_zero, h_inv_partial_g_con)]
    model_g.load_params(new_param)
    h_inv_partial_complex = compute_s_test(model_f, _LoaderF())

    result = tuple([x+(y-z) for x,y,z in zip(model_f.params, h_inv_partial_f, h_inv_partial_complex)])
    return result


def compute_concept_level_influence(
    model_x2c: torch.nn.Module,
    model_c2y: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    masked_concept_idx: int,
    weight_decay: float | None = None,
) -> tuple[torch.Tensor]:
    class _LoaderG:
        @torch.no_grad
        def __iter__(self):
            for data in data_loader:
                data['gt'] = data.pop('concept')
                assert len(data['gt'].shape) == 2 # One-hot encoded
                data['gt'][:, masked_concept_idx] = torch.nan
                yield data
    class _LoaderF:
        @torch.no_grad
        def __iter__(self):
            for data in data_loader:
                x = data['inputs']
                concept_inputs = model_x2c.forward(x)
                concept_inputs = torch.cat(concept_inputs, -1)
                yield {
                    'inputs': concept_inputs,
                    'gt': data['gt'],
                }

    model_g = ModelGradWrap(model_x2c, loss_fn, weight_decay=weight_decay)
    model_f = ModelGradWrap(model_c2y, loss_fn, weight_decay=weight_decay)
    h_inv_partial_g = compute_s_test(model_g, _LoaderG())
    
    new_param = [(x-y) for x,y in zip(model_g.params, h_inv_partial_g)]
    model_g.load_params(new_param)
    h_inv_partial_f = compute_s_test(model_f, _LoaderF())

    result = tuple([x-y for x,y in zip(model_f.params, h_inv_partial_f)])
    return result


def compute_data_level_influence(
    model_x2c: torch.nn.Module,
    model_c2y: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    removed_data: dict[str, torch.Tensor],
    weight_decay: float | None = None,
):
    class _LoaderG:
        @torch.no_grad
        def __iter__(self):
            for data in data_loader:
                con = data['index'] != removed_data['index']
                for k in data: data[k] = data[k][con]
                data['gt'] = data.pop('concept')
                yield data
    class _LoaderF:
        @torch.no_grad
        def __iter__(self):
            for data in data_loader:
                con = data['index'] != removed_data['index']
                for k in data: data[k] = data[k][con]
                x = data['inputs']
                concept_inputs = model_x2c.forward(x)
                concept_inputs = torch.cat(concept_inputs, -1)
                yield {
                    'inputs': concept_inputs,
                    'gt': data['gt'],
                }

    model_g = ModelGradWrap(model_x2c, loss_fn, weight_decay=weight_decay)
    model_f = ModelGradWrap(model_c2y, loss_fn, weight_decay=weight_decay)

    removed_data_inputs = removed_data.pop('inputs')
    removed_data['inputs'] = torch.cat(model_x2c.forward(removed_data_inputs), -1)
    h_inv_partial_f = compute_s_test(model_f, _LoaderF(), removed_data)

    removed_data['inputs'] = removed_data_inputs
    removed_data['gt'] = removed_data.pop('concept')
    h_inv_partial_g = compute_s_test(model_g, _LoaderG(), removed_data)

    new_param = [(x+y) for x,y in zip(model_f.params, h_inv_partial_f)]
    model_f.load_params(new_param)
    b = compute_s_test(model_f, _LoaderF())
    
    new_param = [(x-y) for x,y in zip(model_g.params, h_inv_partial_g)]
    model_g.load_params(new_param)
    a = compute_s_test(model_f, _LoaderF())
    result = tuple([x-y+z for x,y,z in zip(a, b, h_inv_partial_f)])
    return result


def main(args: argparse.Namespace) -> None:
    device = torch.tensor(0).device
    criterion = torch.nn.CrossEntropyLoss()

    loader = load_data(
        [str(Path(args.dataset) / 'test.pkl')], batch_size=args.batch_size,
        use_attr=True, no_img=False, image_dir='images', n_class_attr=112
    )
    
    class _Loader:
        @torch.no_grad
        def __iter__(self):
            for img, class_label, attr_label, idx in loader:
                attr_label = torch.stack(attr_label, 0).T.float()
                class_label = torch.nn.functional.one_hot(class_label, 200).float()
                yield {
                    'inputs': img.to(device),
                    'concept': attr_label,
                    'gt': class_label,
                    'index': idx,
                }

    sample_data = next(iter(_Loader()))
    sample_data = {k:v[:1] for k, v in sample_data.items()}

    
    if 'single' in args.levels:
        model_x2c = torch.load(args.model_x2c).eval().to(device)
        model_c2y = torch.load(args.model_c2y).eval().to(device)
        model_x2c.use_sigmoid = model_c2y.use_sigmoid = True
        compute_single_level_influence_v2(model_x2c, model_c2y, criterion, _Loader(), sample_data)
    
    if 'concept' in args.levels:
        model_x2c = torch.load(args.model_x2c).eval().to(device)
        model_c2y = torch.load(args.model_c2y).eval().to(device)
        model_x2c.use_sigmoid = model_c2y.use_sigmoid = True
        compute_concept_level_influence(model_x2c, model_c2y, criterion, _Loader(), 0)

    if 'data' in args.levels:
        model_x2c = torch.load(args.model_x2c).eval().to(device)
        model_c2y = torch.load(args.model_c2y).eval().to(device)
        model_x2c.use_sigmoid = model_c2y.use_sigmoid = True
        compute_data_level_influence(model_x2c, model_c2y, criterion, _Loader(), sample_data)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model_x2c', type=str)
    parser.add_argument('model_c2y', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--levels', type=str, nargs='+', choices='single,concept,data'.split(','))

    args = parser.parse_args()
    args.levels = set(args.levels or set())
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.set_default_device('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main(args)

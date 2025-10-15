import warnings
from typing import Optional, Sequence, Tuple, Union, List, Any, Dict, Mapping

import torch
import numpy as np
import mmengine
import mmcv
from mmengine.structures import BaseDataElement, PixelData
import mmengine.fileio as fileio
import torch.nn as nn
import torch.nn.functional as F

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if isinstance(input, torch.Tensor):
        resized_mask = F.interpolate(input, size, scale_factor,
                                     mode, align_corners)
    else:
        # 输入为numpy
        resized_mask = F.interpolate(torch.from_numpy(input), size, scale_factor,
                                 mode, align_corners)

    return resized_mask


class SegDataSample(BaseDataElement):

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg

    @property
    def seg_logits(self) -> PixelData:
        return self._seg_logits

    @seg_logits.setter
    def seg_logits(self, value: PixelData) -> None:
        self.set_field(value, '_seg_logits', dtype=PixelData)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self._seg_logits

Number = Union[int, float]
SampleList = Sequence[SegDataSample]
def stack_batch(inputs: List[torch.Tensor],
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len({tensor.ndim for tensor in inputs}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    padded_inputs = []
    padded_samples = []
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            pad_shape = None
            if 'gt_sem_seg' in data_sample:
                gt_sem_seg = data_sample.gt_sem_seg.data
                del data_sample.gt_sem_seg.data
                data_sample.gt_sem_seg.data = F.pad(
                    gt_sem_seg, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_sem_seg.shape
            if 'gt_edge_map' in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(
                    gt_edge_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_edge_map.shape
            if 'gt_depth_map' in data_sample:
                gt_depth_map = data_sample.gt_depth_map.data
                del data_sample.gt_depth_map.data
                data_sample.gt_depth_map.data = F.pad(
                    gt_depth_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_depth_map.shape
            data_sample.set_metainfo({
                'img_shape': tensor.shape[-2:],
                'pad_shape': pad_shape,
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(
                    img_padding_size=padding_size,
                    pad_shape=pad_img.shape[-2:]))

    return torch.stack(padded_inputs, dim=0), padded_samples

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str,
                 None]
class SegDataPreProcessor(nn.Module):

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self._non_blocking = False
        self.device = torch.device('cpu')
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def cast_data(self, data: CastData) -> CastData:
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, '_fields'):
            # namedtuple
            return type(data)(*(self.cast_data(sample) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, (torch.Tensor, BaseDataElement)):
            return data.to(self.device, non_blocking=self._non_blocking)
        else:
            return data

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)
def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)

class LoadImageFromNDArray:

    def __call__(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

class Resize:
    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation='bilinear') -> None:
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes'] * np.tile(
                np.array(results['scale_factor']), 2)
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0,
                                          results['img_shape'][1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['img_shape'][0])
            results['gt_bboxes'] = bboxes

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map'] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['img_shape'][0])
            results['gt_keypoints'] = keypoints

    def __call__(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        return results

class PackSegInputs:
    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def __call__(self, results: dict) -> dict:
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results



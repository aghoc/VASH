import pytest
import torch

from vagt_orca.vision.multiview import predict_views


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA-only inference')
def test_crop_coordinates_deduplication_and_protected_classes():
    calls = 0

    def infer(images):
        nonlocal calls
        result = images.new_zeros((len(images), 300, 6))
        if calls == 0:
            result[:, 0] = images.new_tensor([224, 224, 320, 320, 0.7, 7])
            result[:, 1] = images.new_tensor([10, 20, 80, 90, 0.8, 1])
        else:
            result[:, 0] = images.new_tensor([160, 160, 320, 320, 0.9, 7])
            result[:, 1] = images.new_tensor([10, 20, 80, 90, 0.99, 1])
            result[:, 2] = images.new_tensor([0, 100, 100, 200, 0.99, 10])
        calls += 1
        return result

    result = predict_views(infer, torch.zeros((2, 3, 640, 640), device='cuda'), 'center')
    assert calls == 2
    for batch in result:
        positive = batch[batch[:, 4] > 0]
        assert len(positive) == 2
        torch.testing.assert_close(positive[0], positive.new_tensor([224, 224, 320, 320, 0.9, 7]))
        torch.testing.assert_close(positive[1], positive.new_tensor([10, 20, 80, 90, 0.8, 1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA-only inference')
def test_single_profile_preserves_native_predictions():
    images = torch.zeros((1, 3, 640, 640), device='cuda')
    original = torch.rand((1, 300, 6), device='cuda')
    assert predict_views(lambda frame: original, images, 'single') is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA-only inference')
def test_specialist_routing_preserves_pen_and_non_focus_classes():
    from vagt_orca.vision.focus_profile import merge_specialist

    primary = torch.zeros((1, 300, 6), device='cuda')
    extra = torch.zeros_like(primary)
    primary[0, :3] = primary.new_tensor([[0, 0, 10, 10, .8, 0], [5, 5, 20, 20, .7, 7], [10, 10, 40, 40, .6, 13]])
    extra[0, :2] = primary.new_tensor([[1, 1, 11, 11, .9, 0], [50, 50, 60, 60, .99, 7]])
    for strategy in ('route', 'union'):
        result = merge_specialist(primary, extra, (0,), strategy)
        positive = result[0, result[0, :, 4] > 0]
        assert positive[:, 5].tolist() == [0, 7, 13]
        torch.testing.assert_close(positive[0], extra[0, 0])
        torch.testing.assert_close(positive[1:], primary[0, 1:3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA-only inference')
def test_small_toothbrush_retention_does_not_restore_large_primary_false_positive():
    from vagt_orca.vision.focus_profile import merge_specialist

    primary = torch.zeros((1, 300, 6), device='cuda')
    extra = torch.zeros_like(primary)
    primary[0, :2] = primary.new_tensor([[0, 0, 8, 16, .7, 10], [100, 100, 200, 300, .99, 10]])
    extra[0, 0] = primary.new_tensor([300, 300, 400, 500, .8, 10])
    result = merge_specialist(primary, extra, (10,), small_primary_size=32)
    positive = result[0, result[0, :, 4] > 0]
    assert len(positive) == 2
    torch.testing.assert_close(positive[0], extra[0, 0])
    torch.testing.assert_close(positive[1], primary[0, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA-only inference')
def test_full_view_survives_a_backend_reusing_its_output_buffer():
    images = torch.zeros((1, 3, 640, 640), device='cuda')
    buffer = images.new_zeros((1, 300, 6))
    calls = 0

    def infer(frame):
        nonlocal calls
        buffer.zero_()
        if calls == 0:
            buffer[0, 0] = buffer.new_tensor([10, 20, 80, 90, .8, 1])
        else:
            buffer[0, 0] = buffer.new_tensor([160, 160, 320, 320, .9, 7])
        calls += 1
        return buffer

    result = predict_views(infer, images, 'center')
    positive = result[0, result[0, :, 4] > 0]
    assert positive[:, 5].tolist() == [7, 1]
    torch.testing.assert_close(positive[0], images.new_tensor([224, 224, 320, 320, .9, 7]))
    torch.testing.assert_close(positive[1], images.new_tensor([10, 20, 80, 90, .8, 1]))

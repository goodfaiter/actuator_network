import torch

from actuator_network.helpers.pandas_to_torch import process_inputs_time_series, process_outputs_time_series


def test_process_inputs_time_series_shape_and_forward_window():
    data = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]
    )
    history_size = 3
    stride = 1

    result = process_inputs_time_series(data, history_size=history_size, stride=stride, prediction=False)

    assert result.shape == (3, 3, 2)

    # Forward-looking window: result[i, j] should equal data[i + j * stride]
    assert torch.allclose(result[0], data[:3])
    assert torch.allclose(result[1], data[1:4])
    assert torch.allclose(result[2], data[2:5])


def test_process_inputs_time_series_stride_and_drops_incomplete():
    data = torch.tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ]
    )
    history_size = 3
    stride = 2

    result = process_inputs_time_series(data, history_size=history_size, stride=stride, prediction=False)

    # Need indices 0, 2, 4 -> batch_size 4 cannot provide index 4, so no valid sequences
    assert result.shape == (0, 3, 1)


def test_process_outputs_time_series_matches_last_input_index():
    data = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]
    )
    history_size = 3
    stride = 1

    inputs = process_inputs_time_series(data, history_size=history_size, stride=stride, prediction=False)
    outputs = process_outputs_time_series(data, history_size=history_size, stride=stride)

    assert outputs.shape == (3, 1, 2)
    assert inputs.shape[0] == outputs.shape[0]

    # Output for sample i should be the last element of inputs[i]
    expected = inputs[:, -1, :].unsqueeze(1)
    assert torch.allclose(outputs, expected)

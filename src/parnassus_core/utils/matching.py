import numpy as np
from scipy.optimize import linear_sum_assignment


def reshape_phi(phi):
    """
    Reshape phi values to ensure they are in the range [-pi, pi].
    This is necessary for correct distance calculations in phi.
    """
    return np.atan2(np.sin(phi), np.cos(phi))


def match_objects(
    input_eta, input_phi, truth_eta, truth_phi, dr_cut=0.6, pad_to_len=None
):
    truth_eta = np.tile(
        np.expand_dims(truth_eta, axis=1), (1, len(input_eta))
    )  # row content same
    input_eta = np.tile(
        np.expand_dims(input_eta, axis=0), (len(truth_eta), 1)
    )  # column content same

    truth_phi = np.tile(
        np.expand_dims(truth_phi, axis=1), (1, len(input_phi))
    )  # row content same
    input_phi = np.tile(
        np.expand_dims(input_phi, axis=0), (len(truth_phi), 1)
    )  # column content same

    loss_phi = np.pow(reshape_phi(truth_phi - input_phi), 2)
    loss_eta = np.power(truth_eta - input_eta, 2)

    loss = loss_eta + loss_phi

    dr = np.sqrt(loss_eta + loss_phi)
    loss[dr > dr_cut] = 1e3

    loss[loss == np.inf] = 1000
    loss[loss == np.nan] = 1000
    truth_ix, input_ix = linear_sum_assignment(loss)

    # Create boolean mask for dr < 0.6
    mask = dr <= dr_cut

    # Find all pairs (i, j) where dr < 0.6
    filtered_pairs = np.argwhere(mask)

    # Convert truth_ix and input_ix into sets for fast lookups
    truth_input_pairs = set(zip(truth_ix, input_ix))

    # Select new pairs where (i, j) are in the filtered_pairs and truth_input_pairs
    new_pairs = np.array(
        [pair for pair in filtered_pairs if tuple(pair) in truth_input_pairs]
    )
    # Split new pairs into new_truth_ix and new_input_ix
    if len(new_pairs) == 0:
        if pad_to_len is not None:
            new_truth_ix = np.ones(pad_to_len, dtype=int) * -1
            new_input_ix = np.arange(pad_to_len)
        else:
            new_truth_ix = []
            new_input_ix = []
    else:
        new_truth_ix, new_input_ix = new_pairs.T

    input_ix = np.array(new_input_ix)
    truth_ix = np.array(new_truth_ix)

    if len(new_pairs) == 0:
        hung_cost = 1000
    else:
        indices = input_ix.shape[0] * truth_ix + input_ix
        loss_extract = np.take_along_axis(loss.flatten(), indices, axis=0)
        hung_cost = loss_extract.mean()
    if len(input_ix) > 0 and pad_to_len is None:
        assert np.max(input_ix) < input_eta.shape[-1], (
            f"{mask.shape=} {input_ix=} {len(input_eta)=}"
        )
    if len(truth_ix) > 0 and pad_to_len is None:
        assert np.max(truth_ix) < truth_eta.shape[0], (
            f"{mask.shape=} {truth_ix=} {len(truth_eta)=}"
        )
    if pad_to_len is not None:
        if len(input_ix) < pad_to_len:
            input_ix = np.pad(
                input_ix, (0, pad_to_len - len(input_ix)), constant_values=-1
            )
        if len(truth_ix) < pad_to_len:
            truth_ix = np.pad(
                truth_ix, (0, pad_to_len - len(truth_ix)), constant_values=-1
            )

    return input_ix, truth_ix, hung_cost

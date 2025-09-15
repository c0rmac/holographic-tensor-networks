# In bp.py

import quimb as qu
import quimb.tensor as qtn
import numpy as np
import copy


def belief_propagation(
        tn,
        max_iterations=100,
        tol=1e-6,
        damping=0.0,
        verbose=True,
):
    """
    Performs Belief Propagation on an arbitrary tensor network.

    Args:
        tn (qtn.TensorNetwork): The tensor network to contract.
        max_iterations (int): Maximum number of BP iterations.
        tol (float): Convergence tolerance for message difference.
        damping (float): Damping factor (0 to 1) to stabilize convergence.
        verbose (bool): Whether to print convergence information.

    Returns:
        tuple[dict, bool]: A tuple containing:
            - The final dictionary of converged message tensors.
            - A boolean flag indicating if the algorithm converged.
    """
    # 1. Initialize messages
    messages = {}
    for ix in tn.inner_inds():
        tids = tn.ind_map[ix]
        if len(tids) != 2:
            raise ValueError(f"Index '{ix}' is not shared by exactly two tensors.")
        tid_a, tid_b = tids

        dim = tn.ind_size(ix)
        # The initial message should be a rank-1 vector, not a rank-2 matrix.
        # A uniform vector of ones is a standard unbiased initialization.
        initial_msg_data = np.ones(dim, dtype=complex)
        message_tensor = qtn.Tensor(initial_msg_data, inds=(ix,))

        # Message keys are now consistently the internal integer tensor IDs
        messages[(tid_a, tid_b)] = message_tensor
        messages[(tid_b, tid_a)] = message_tensor.copy()

    # --- Main BP Loop ---
    for i in range(max_iterations):
        old_messages = copy.deepcopy(messages)
        new_messages = {}
        max_rel_diff = 0.0

        # Iterate over each tensor object in the network
        for tid_a, t_a in tn.tensor_map.items():

            # Find all neighbors of the current tensor
            neighbor_tids = tn._get_neighbor_tids(tid_a)

            # For each neighbor, compute the message from t_a to it
            for tid_b in neighbor_tids:

                # Gather incoming messages to t_a, EXCLUDING the one from t_b
                tensors_to_contract = [t_a]
                for tid_c in neighbor_tids:
                    if tid_c != tid_b:
                        # Look up message using robust integer IDs
                        tensors_to_contract.append(old_messages[(tid_c, tid_a)])

                # Find the index connecting t_a and t_b to use as output
                # This is a robust way to get the shared index
                t_b = tn.tensor_map[tid_b]
                common_ix = set(t_a.inds) & set(t_b.inds)
                output_ix = tuple(common_ix)

                # --- FIX IS HERE ---
                # Create a temporary TensorNetwork and call its .contract() method.
                sub_tn = qtn.TensorNetwork(tensors_to_contract)
                new_msg = sub_tn.contract(all, output_inds=output_ix)

                if damping > 0:
                    new_msg = (1 - damping) * new_msg + damping * old_messages[(tid_a, tid_b)]

                new_messages[(tid_a, tid_b)] = new_msg

        # --- Check for convergence ---
        for edge in messages.keys():
            new_msg = new_messages[edge]
            old_msg = old_messages[edge]

            diff_norm = (new_msg - old_msg).norm()
            new_norm = new_msg.norm()

            if new_norm > 1e-12:
                rel_diff = diff_norm / new_norm
                if rel_diff > max_rel_diff:
                    max_rel_diff = rel_diff

        messages = new_messages

        if verbose:
            print(f"Iteration {i + 1}/{max_iterations}, Max Relative Difference: {max_rel_diff:.4e}")

        if max_rel_diff < tol:
            if verbose:
                print(f"\n✅ Converged in {i + 1} iterations.")
            return messages, True

    print(f"\n⚠️ Warning: BP did not converge within {max_iterations} iterations.")
    return messages, False


def get_belief(tn, tensor_id, final_messages):
    """
    Computes the local belief (marginal) for a specific tensor using its ID.

    Args:
        tn (qtn.TensorNetwork): The original tensor network.
        tensor_id (int): The internal ID of the tensor to compute the belief for.
        final_messages (dict): The converged messages from the BP algorithm.

    Returns:
        qtn.Tensor: The belief tensor for the specified tensor.
    """
    # --- FIX IS HERE ---
    # Handle both string tags and integer IDs as input
    if isinstance(tensor_id, str):
        # If a string tag is given, find the corresponding internal integer ID.
        # This assumes the tag uniquely identifies one tensor.
        try:
            # quimb provides a method to get TIDs from tags
            internal_tid = list(tn._get_tids_from_tags(tensor_id))[0]
        except IndexError:
            raise KeyError(f"No tensor found with unique tag '{tensor_id}'")
    else:
        # If an integer is given, assume it's the internal ID
        internal_tid = tensor_id

    t_a = tn.tensor_map[internal_tid]

    neighbor_tids = tn._get_neighbor_tids(internal_tid)

    tensors_to_contract = [t_a]
    for neighbor_tid in neighbor_tids:
        tensors_to_contract.append(final_messages[(neighbor_tid, internal_tid)])

    sub_tn = qtn.TensorNetwork(tensors_to_contract)
    belief = sub_tn.contract(all)
    return belief

def contract_with_bp(
        tn: qtn.TensorNetwork,
        max_iterations: int = 50,
        tol: float = 1e-7,
) -> qtn.Tensor:
    """
    Approximately contracts a tensor network to a single tensor using
    Belief Propagation (BP).

    Args:
        tn: The quimb TensorNetwork to contract.
        max_iterations: The maximum number of iterations for the BP algorithm.
        tol: The convergence tolerance for the BP algorithm.

    Returns:
        The final, single quimb.Tensor representing the contracted network.
    """
    # 1. Run the Belief Propagation algorithm to get converged messages
    final_messages, converged = belief_propagation(
        tn, max_iterations=max_iterations, tol=tol
    )

    if not converged:
        print("Warning: BP contractor did not converge.")

    # 2. Compute the final result by calculating the belief at any single tensor
    #    and contracting it. The result is independent of the tensor chosen.
    any_tid = next(iter(tn.tensor_map.keys()))

    # The get_belief function contracts the chosen tensor with all its final
    # incoming messages, yielding the final approximate tensor.
    result_tensor = get_belief(tn, any_tid, final_messages)

    return result_tensor
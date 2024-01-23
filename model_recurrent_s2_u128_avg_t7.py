import tensorflow as tf
import util
from functools import partial
from tensorflow.keras import layers

# Updated - unroll for eager execution and removing calls unnecesary for tensorflow v2
# TODO: Might need to be updated again to get rid of reuse input

def _unroll_rnn(x, x_name, T, rnn_graph, average_on_relu, reuse):
    """
    Construct the TF graph for an RNN defined by its nodes and edges.

    Input:
        x: the input tensor
        rnn_graph: [node_names, edges]
            edges are tuples of <from_node_name, to_node_name, edge_func, delay>.
        average_on_relu: if true, average instead of sum will be used
            to combine outputs from multiple edges.
        reuse: reuse flag for sharing variables.

    Output:
        all tensors: the output tensor at every time step.
    """
    node_names, edges = rnn_graph
    all_tensors = [{} for _ in range(T)]
    
    # Counting the number of input tensors of a node.
    # only useful if average_on_relu=True.
    counter = [{} for _ in range(T)]
    print(type(x),tf.shape(x))
    all_tensors[0][x_name] = x
    counter[0][x_name] = 1

    for t in range(T):
        # Unroll at time step t.
        for s_name in all_tensors[t]:
            # Finalize state, apply relu and scale.
            in_tensor = all_tensors[t][s_name]
            in_tensor = tf.keras.activations.relu(in_tensor)
            if average_on_relu and counter[t][s_name] > 1:
                in_tensor = in_tensor / counter[t][s_name]
            all_tensors[t][s_name] = in_tensor

            # Look at s_name, find all its successors.
            for edge in edges:
                e_name1, e_name2, e_fun, e_delay = edge
                if s_name == e_name1:
                    if e_delay + t < T:
                        try:
                            out_tensor = e_fun(in_tensor, reuse=reuse)

                        except ValueError:
                            # Sharing if exists.
                            out_tensor = e_fun(in_tensor, reuse=False)

                        if e_name2 in all_tensors[e_delay + t]:
                            all_tensors[e_delay +
                                        t][e_name2] = all_tensors[e_delay +
                                                                  t][e_name2] + out_tensor
                            counter[e_delay + t][e_name2] += 1
                        else:
                            all_tensors[e_delay + t][e_name2] = out_tensor
                            counter[e_delay + t][e_name2] = 1
    return all_tensors

# Updated - Upsample function using keras layers initialization
def _conv_upsample(_input, hidden_size, reuse):
    """Conv layer that upsamples the input."""
    output = tf.keras.layers.Conv2DTranspose(
        filters=hidden_size, # alteration to avoid mismatched final dimension
        kernel_size=2,
        strides=2,
        activation=None)(_input)
    return output


# Updated - nn.avg_pool replaced by nn.avg_pool2d, function remains the same
def _downsample(_input, reuse):
    """Down-sample via average pooling."""
    output = tf.nn.avg_pool2d(
        _input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return output

# Updated - conv layers initialized using Keras and removal of reuse logic
def _conv_link(_input, name, hidden_size, reuse):
    _tmp = tf.keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=3,
        padding='SAME',
        activation=tf.keras.activations.relu,
    )(_input)
    output = tf.keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=3,
        padding='SAME',
        activation=None,
    )(_tmp)
    return output

# No need to update
def _skip_link(_input, reuse):
    return _input

# Updated - Layer activation using Keras, updated summing functions and other updated utility functions
def build_model(scale, training, reuse):
    hidden_size = 128
    use_average_out = True
    T = 7

    input_tensor = layers.Input(shape=(None, None, 3))

    # Define the RNN in its recurrent form.
    # s1: low-res state. s2: high-res state.
    node_names = ['s1', 's2']
    edges = []
    edges.append(['s1', 's1', _skip_link, 1])
    edges.append(['s1', 's1', partial(_conv_link, hidden_size=hidden_size, name='s1s1'), 1])
    edges.append(['s1', 's2', partial(_conv_upsample, hidden_size=hidden_size), 1])
    edges.append(['s2', 's1', partial(_downsample), 1])
    edges.append(['s2', 's2', _skip_link, 1])
    edges.append(['s2', 's2', partial(_conv_link, hidden_size=hidden_size, name='s2s2'), 1])    

    
    s1_0 = layers.Conv2D(
        filters=hidden_size / 2,
        kernel_size=3,
        activation=tf.keras.activations.relu,
        name='in_0',
        padding='same')(input_tensor)
    
    s1_1 = layers.Conv2D(
        filters=hidden_size,
        kernel_size=3,
        activation=tf.keras.activations.relu,
        name='in_1',
        padding='same')(s1_0)

    full_net_states = _unroll_rnn(s1_1, 's1', T, (node_names, edges), average_on_relu=False, reuse=reuse)
    
    if use_average_out:
        all_out_states = [
          full_net_states[t]['s2'] for t in range(T)
          if 's2' in full_net_states[t]
        ]                
        pre_out = tf.add_n(all_out_states) / len(all_out_states)
    else:
        pre_out = full_net_states[T - 1]['s2']

    # Final output prediction.
    out = layers.Conv2D(filters=3, kernel_size=1, activation=None, name='out')(pre_out)

    model = tf.keras.Model(inputs=input_tensor, outputs=out)

    return model

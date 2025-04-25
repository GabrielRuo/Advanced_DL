
import torch
import torch.nn as nn
import torch_geometric.nn as graphnn
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data


def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None
):#-> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    COMPLETE

    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    # Assumptions (remove it for the bonus)
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    c,h,w = image.shape
    kernel_size = conv2d.kernel_size[0]
#concatenate the width and length and have the features for each node be the values at a point (i,j) for all channels
    x = image.view(c,-1).T
# Create coordinates for each node
    coords = []
    for r_ in range(h):
        for col_ in range(w):
            coords.append([col_, -r_])  # (x, y)
    pos = torch.tensor(coords, dtype=torch.float)
    radius = (kernel_size - 1)//2
    # connect close neighbors
    edge_index = radius_graph(pos, r=radius+0.5)

    #each edge source --> targe has as attributes the coordinates of the the vector source - target to later match it with the kernel
    edge_attr = pos[edge_index[0]] - pos[edge_index[1]]

    #shift so that attributes correspond to indices
    def edge_attr_to_kernel_indices(edge_attr: torch.Tensor, kernel_size: int) -> torch.Tensor:
      """
      Converts edge attributes [x, y] from spatial coordinates (right, up)
      into kernel indices [row, col] suitable for indexing into a
      (kernel_size x kernel_size) tensor.
      """
      center = (kernel_size - 1) // 2
      x = edge_attr[:, 0]
      y = edge_attr[:, 1]
      row = center - y  # up → row decreases
      col = center + x  # right → col increases
      return torch.stack([row, col], dim=1).long()

    edge_attr = edge_attr_to_kernel_indices(edge_attr, kernel_size)

    #Add self-loops to the adjacency matrix and corresponding attributes
    def add_self_loops_with_attrs(edge_index, edge_attr, num_nodes, radius):
      # Add self-loop edges
      self_loops = torch.arange(num_nodes, dtype=torch.long)
      self_loop_edges = torch.stack([self_loops, self_loops], dim=0)  # shape [2, num_nodes]

      # Add matching self-loop attributes: center of the kernel
      self_loop_attrs = torch.full((num_nodes, 2), radius, dtype=torch.long)  # shape [num_nodes, 2]

      # Concatenate with existing edge_index and edge_attr
      edge_index = torch.cat([edge_index, self_loop_edges], dim=1)
      edge_attr = torch.cat([edge_attr, self_loop_attrs], dim=0)
      return edge_index, edge_attr

    edge_index, edge_attr = add_self_loops_with_attrs(edge_index, edge_attr, x.size(0), radius)

    #create graph object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    # Assumptions (remove it for the bonus)
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."
    h,w = height,width
    _,c = data.shape #c = number of out_channels
    image = (data.T).reshape(c,h,w)
    return image



class Conv2dMessagePassing(graphnn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        super().__init__(aggr = 'add')
        self.kernel_size = (conv2d.kernel_size)[0]
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        kernel = conv2d.weight.data.clone()#kernel shape [out_c, in_c, kH, kW]

        # Register as a learnable parameter
        self.kernel = nn.Parameter(kernel)

    def forward(self, data):
        self.edge_index = data.edge_index
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph indexed by i,
        the message trough the edge e (ie from node u to node v)
        should be returned as the i-th line of the output tensor.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the source node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size E x out_channels)
        """
        # The sum in the message is over the channels as we created a graph which combined all the channels into a single point.
        # The aggregate function will then sum over the source nodes for each target node.

        #invert shape to match kernel shape and add out_channel dimension
        x_j = x_j.T
        x_j = x_j.unsqueeze(0)

        #multiply the source nodes with associated kernel value
        kernel_values = self.kernel[:,:,edge_attr[:,0],edge_attr[:,1]]
        kernel_times_pixels_per_in_channel = kernel_values*x_j

        #sum over the input channels
        kernel_times_pixels = torch.sum(kernel_times_pixels_per_in_channel, axis = 1)

        #invert shape again to match expected x shape
        return kernel_times_pixels.T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as graphnn

class StudentModel(nn.Module):
    def __init__(self):
      super(StudentModel, self).__init__()#attention à l'initialisation de Gloriot
      self.GAT1 = graphnn.conv.GATConv(in_channels = n_features, out_channels = 256, heads = 4, concat = True)
      self.GAT2 = graphnn.conv.GATConv(in_channels = 1024, out_channels = 256, heads = 4, concat=True)
      self.GAT3 = graphnn.conv.GATConv(in_channels = 1024, out_channels = 121, heads = 6, concat=False) #concat = False averages
      self.linear_skip = nn.Linear(1024, 1024)
    #init is defautl Glorot in pytorch

    def forward(self, x, edge_index):
        x = self.GAT1(x, edge_index)
        x = F.elu(x)
        h = self.GAT2(x, edge_index) # skip connection suggested by the article
        x = h + self.linear_skip(x) #tried this to enrich the skip connection 
        x = F.elu(x) # important to have non linearity after the skip connection (increase in F1 score by 3 points)
        x = self.GAT3(x, edge_index)
        #no need for sigmoid as automatically computed by the loss
        return x
    
# Initialize model
model = StudentModel()

## Save the model
torch.save(model.state_dict(), "model.pth")

### This is the part we will run in the inference to grade your model
## Load the model
model = StudentModel()  # !  Important : No argument
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
print("Model loaded successfully")
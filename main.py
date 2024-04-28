from dependencies import *
from Net import FullyConnectedNetworkMod
from Trainer import Trainer
from Validator import Validator
from Loss import *
from Loss_PINN import *
from Utils import *
# Check if GPU is availabls

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


##Model Parameters
u_range=(-.1,.81)
v_range=(0,.12)
k_range=(0,1)
t_range=(0,50)

##Model Arch
input_shape = 4  
output_shape = 2   
hidden_layer = [(nn.Tanh,64), (nn.ReLU,32),(nn.ReLU,64),(nn.Tanh,32)]
model = FullyConnectedNetworkMod(input_shape, output_shape, hidden_layer).to(device)
trainer=Trainer(model)

print(model)




##DataLoss
data_folder="training_data/treino/"



trainer.add_loss(LOSS.fromDataSet(data_folder,1024,device=device,loss_type="L4"),weigth=10)
trainer.add_loss(LOSS.fromDataSet(data_folder,1024,device=device,loss_type="L2"),weigth=1)


##LOSS_PINN

def FHN_LOSS(data_in, model):
        x,w = model(data_in).T 
        t,u,v,k=data_in.T

        dx_dt= torch.autograd.grad(
        x,
        data_in,
        grad_outputs=torch.ones_like(t),
        create_graph=True,
        retain_graph=True)[0].T[0]
        
        dw_dt= torch.autograd.grad(
        x,
        data_in,
        grad_outputs=torch.ones_like(t),
        create_graph=True,
        retain_graph=True)[0].T[0]
        
        
        pde = torch.abs( 10*((k + 1)*(u*(u-0.4)*(1-u))-v ) - dx_dt)
        return pde

batch_gen=lambda size:default_batch_generator(size,[t_range,u_range,v_range,k_range])

trainer.add_loss(LOSS_PINN(FHN_LOSS,batch_gen,batch_size=1024),weigth=10)



##BoundaryLoss

def f(data_in, model):
        x,w = model(data_in).T 
        t,u,v,k=data_in.T          
        return torch.abs(x-u) + torch.abs(w-v)


batch_gen=lambda size:default_batch_generator(size,[(0,0),u_range,v_range,k_range])
trainer.add_loss(LOSS_PINN(f,batch_gen),weigth=1000)


##Validator
trainer.add_validator(Validator.fromDataSet("training_data/validation/",device=device))











trainer.train(100000)

print(model)

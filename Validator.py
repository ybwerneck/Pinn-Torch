from .dependencies import *
def ensure_at_least_one_column(x):
    # If x is 1-dimensional, reshape to (len(x), 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    # If x is already 2-dimensional, return it unchanged
    elif x.ndim == 2:
        return x
    else:
        raise ValueError("Input array must be 1D or 2D")
        

class Validator():
    def __init__(self,data_in,target,name="val",device=torch.device("cuda"),dump_f=0,dump_func=0,script=0):
        self.data_in=data_in.to("cpu")
     #   print(self.data_in)
        self.device=device
        self.dump_f=dump_f
    #    print("\n\n",dump_func)
        self.dump_func=self.dump_f_def if  dump_func==0 else lambda dump:dump_func(self,dump=dump) 
        self.target=target.to("cpu")
        self.name=name
        self.count=0
        self.last_out=0
        self.model=0
        self.script=script
        
    def setFolder(self,folder):
        self.folder=folder
        h5py.File(f"{self.folder}/{self.name}_err.h5", 'w')
    

    def dump_f_def(self,target=0,out=0,data_in=0,sufix="",dump=False):       
       # print(dump)
        try:
            print(target,out)
            if(target==0):
                target=self.target
            if(out==0):
                out=self.last_out
            if(data_in==0):
                data_in=self.data_in
        except:
            a=0
        mean_error = torch.mean(torch.abs(out-target)).detach()
        max_error =torch.max(torch.abs(out-target)).detach()
       # print("Validation error: ", mean_error.item(), max_error.item())
        new_data = np.array([mean_error.detach().cpu(), max_error.detach().cpu()])
        # Write data to HDF5 file
        
        
        with h5py.File(f"{self.folder}/{self.name}_{sufix}err.h5", 'a') as hf:
            # Check if dataset exists
            if "error_stats" in hf:
                
                error_stats_ds = np.array(hf["error_stats"]).flatten()              
                updated_data = np.zeros(len(error_stats_ds)+2)
                updated_data[:-2] = error_stats_ds[:]
                
                # Add the new data
                updated_data[-2:] = new_data
                updated_data=np.reshape(updated_data,(len(updated_data)//2,2))
                # Replace the existing dataset with the updated data
                del hf["error_stats"]
                hf.create_dataset("error_stats", data=updated_data)
            else:
                # Create a new dataset if it doesn't exist
                hf.create_dataset("error_stats", data=new_data)
        if(dump==True):
            with h5py.File(f"{self.folder}/{self.name}{sufix}.h5", 'w') as hf:
                    hf.create_dataset("target", data=target.detach().cpu().numpy())
                    hf.create_dataset("pred", data=out.detach().cpu().numpy())
                    hf.create_dataset("input", data=data_in.detach().cpu().numpy())
                

    def val(self, model,p=False):
        # Evaluate the model
        self.model=model
        batch_size = 5*2048  # Choose an appropriate batch size
        num_samples = len(self.data_in)
        num_batches = (num_samples + batch_size - 1) // batch_size

        # Initialize variables to store total absolute error and maximum error
        total_error = 0
        data_out=torch.zeros((num_samples,len(self.target.T)),requires_grad=False).to("cpu")
        # Iterate over batches
        with torch.no_grad():
            for i in range(num_batches):

                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                batch_data = self.data_in[start_idx:end_idx]
                batch_target = self.target[start_idx:end_idx]

                # Evaluate the model on the current batch
             #   print(self.device)
                cudad=ensure_at_least_one_column(batch_data.to(self.device))
                #print(np.shape(cudad))

                data_out[start_idx:end_idx]= model(cudad).to("cpu")
                    # Clear GPU memory
                del batch_data
                del batch_target
                torch.cuda.empty_cache()


        # Calculate mean error and maximum error over all batches
        self.last_out=data_out
        dump=(self.count%self.dump_f==0)

        self.dump_func(dump=dump)
    
        self.count+=1

                    
        a=0
        mean_error = torch.mean(torch.abs(data_out-self.target)).detach()
        max_error =torch.max(torch.abs(data_out-self.target)).detach()
        self.dump_func(dump=dump)
    
        self.count+=1
        if(dump):
            if(self.script!=0):
                self.script()


        return mean_error


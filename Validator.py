from .dependencies import *
class Validator():
   
    def __init__(self,data_intarget,target,name="val",device=torch.device("cuda")):
        self.data_in=data_intarget.to("cpu")
        self.device=device
        self.target=target.to("cpu")
        self.name=name
 
    def setFolder(self,folder):
        self.folder=folder
        h5py.File(f"{self.folder}/{self.name}_err.h5", 'w')
    
    def val(self, model,p=False):
        # Evaluate the model
        
        batch_size = 10*2048  # Choose an appropriate batch size
        num_samples = len(self.data_in)
        num_batches = (num_samples + batch_size - 1) // batch_size

        # Initialize variables to store total absolute error and maximum error
        total_error = 0
        max_error = float('-inf')
        data_out=torch.zeros((num_samples,2),requires_grad=False).to("cpu")
        # Iterate over batches
        with torch.no_grad():
            for i in range(num_batches):
                #print(i)
                #print("GPU memory allocated:", torch.cuda.memory_allocated())
                #print("GPU memory cached:", torch.cuda.memory_cached())
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                batch_data = self.data_in[start_idx:end_idx]
                batch_target = self.target[start_idx:end_idx]

                # Evaluate the model on the current batch
                cudad=batch_data.to(self.device)
                data_out[start_idx:end_idx]= model(cudad).to("cpu")
                    # Clear GPU memory
                del batch_data
                del batch_target
                torch.cuda.empty_cache()


        # Calculate mean error and maximum error over all batches
        mean_error = torch.mean(torch.abs(data_out-self.target)).detach()
        max_error =torch.max(torch.abs(data_out-self.target)).detach()
        new_data = np.array([mean_error, max_error])
        # Write data to HDF5 file
        with h5py.File(f"{self.folder}/{self.name}_err.h5", 'a') as hf:
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

        if(p):
            with h5py.File(f"{self.folder}/{self.name}.h5", 'w') as hf:
                hf.create_dataset("target", data=self.target.detach().cpu().numpy())
                hf.create_dataset("pred", data=data_out.detach().cpu().numpy())
                hf.create_dataset("input", data=self.data_in.detach().cpu().numpy())
        return max_error


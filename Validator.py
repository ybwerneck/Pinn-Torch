from dependencies import *
class Validator():
   
    def __init__(self,data_intarget,target,name="val"):
        self.data_in=data_intarget
        self.target=target
        self.name=name
 
    def setFolder(self,folder):
        self.folder=folder
        h5py.File(f"{self.folder}/{self.name}_err.h5", 'w')
                  
    def val(self, model):
        # Evaluate the model
        p = model(self.data_in)
        e = torch.abs((torch.abs(self.target) - torch.abs(p))).detach().cpu().numpy()
        
        # Write data to HDF5 file
        with h5py.File(f"{self.folder}/{self.name}_err.h5", 'a') as hf:
            # Check if dataset exists
            if "error_stats" in hf:
                new_data = np.array([np.mean(e), np.max(e)])
                
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
                hf.create_dataset("error_stats", data=[[np.mean(e), np.max(e)]])

            
        with h5py.File(f"{self.folder}/{self.name}.h5", 'w') as hf:
            #hf.create_dataset("error_stats", data=[np.mean(e), np.max(e)])
            hf.create_dataset("target", data=self.target.detach().cpu().numpy())
            hf.create_dataset("pred", data=p.detach().cpu().numpy())
            hf.create_dataset("input", data=self.data_in.detach().cpu().numpy())


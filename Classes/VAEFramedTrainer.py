import torch
import torch.optim as optim

from .ExperimentConfig import ExperimentConfig
from .VariationalModel import VariationalAutoEncoder
from .ActivationRecorder import ActivationRecorder
from .MI_Estimator import MI_Estimator
from .MI_History import MI_History

from Functions.VAE_info import VAE_info





class VAEFramedTrainer:
    
    def __init__(self, ds_manager, device):
        
        self.device = device
        # Take dataloaders from ds_manager
        self.train_loader = ds_manager.train_loader
        self.test_loader  = ds_manager.test_loader


    # ---------------------------------------LOSS FUNCTION UTILITY -----------------------------------------------
    
    # Setup loss function (b is the quantize z in VAE class)
    def loss_function(self, x, x_hat, z, b, penalize_lambda, premium_lambda, bit_type, mean, logVar, Variational):

        mse = torch.nn.functional.mse_loss(x_hat, x, reduction="sum") # sum over all pixels

        # case where penalize and premium are zero
        if not penalize_lambda:
            penalty = torch.tensor(0, device=self.device)     
        if not premium_lambda:
            premium = torch.tensor(0, device=self.device)

        
        match bit_type:
        
            case "real": # normal, no restiction at all
                penalty = torch.tensor(0, device=self.device)
                premium = torch.tensor(0, device=self.device)
                

            case "restricted": # case z=-1/+1 is implemented     
                #penalty = penalize_lambda * torch.sum( (p*(p-1))**2 ) # penalize values different both from 0 or 1
                if penalize_lambda:
                    penalty = penalize_lambda * torch.sum( (z*z-1)**2 )# penalize values different both from -1 or 1

                # not working because into log(p) we can get negative numbers starting from z < -1
                # remember in resticted case we still have real values for z
                #p = float( torch.mean( (z+1)/2 ) ) # after map -1/+1 into 0/1 we estimate probability of bit=+1
                #premium = premium_lambda * ( p*np.log(p) +(1-p)*np.log(1-p) )  # premium base on - entropy
                
                # simpler version of less penalize more mixed valued of z (namely splitted into -1 and +1)
                if premium_lambda:
                    premium = premium_lambda * torch.sum( (z*z-1)**2 ) # case premium_lambda=0 is already fine

            case "discrete":
                if penalize_lambda:
                    penalty = penalize_lambda * torch.sum( (z - b.detach())**2 ) # to push z near b (values -1 or 1)
                
                premium = torch.tensor(0, device=self.device)
                
            case _: # every other cases
                raise ValueError(f"bit_type must be 'real', 'restricted', or 'discrete', you wrote '{bit_type}' not a valid choice.")

        if Variational:
            kl_loss = -0.5 * torch.sum(1 + logVar - mean.pow(2) - logVar.exp())
        else:
            kl_loss = torch.tensor(0, device=self.device)

        total = mse + kl_loss + penalty + premium

        return mse, kl_loss, penalty, premium, total


    # ---------------------------------------RUN (TRAINING)------------------------------------------------------
    
    def run(self, config: ExperimentConfig, Debug=False, Validation=False):
        print(f"\n{'='*15} STARTING EXPERIMENT: {config.name} {'='*15}")
        
        # _______________________________ INITIALIZATION ___________________________________ 
        # Initialize the model using the parameters dictionary
        model = VariationalAutoEncoder(**config.model_params).to(self.device)

        # Setup loss function
        bit_type = config.model_params.get("bit_type", "real")
        penalize_lambda = config.train_params.get("penalize_lambda", 0)
        premium_lambda = config.train_params.get("premium_lambda", 0)
        Variational = config.model_params.get('Variational', True)
        
        # Setup Optimizer
        lr = config.train_params.get("lr", 1e-3)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Setup Recorder
        recorder = ActivationRecorder()
        recorder.InitialRegister(model)
        recorder.activate_recording(False)

        # Setup MI estimator and storage
        method = config.mi_params.get("method", ["kde"]*6)
        default= config.mi_params.get("default", "kde")
        sigma  = config.mi_params.get("sigma", 1.0)
        n_neig = config.mi_params.get("n_neig", 3)
        mi_estimator = MI_Estimator( method=method, sigma=sigma, n_neig=n_neig, default=default )
        mi_history = MI_History()
        
        epochs = config.train_params.get("epochs", 20)
        model.train()

        # Show only epoch progrosse at 0%,25%,50%,75%,100%
        if epochs <= 5:
            show_epochs = list(range(1, epochs + 1))
        else:
            show_epochs = [
                1,
                int(0.25 * epochs),
                int(0.50 * epochs),
                int(0.75 * epochs),
                epochs
            ]
        show_epochs = sorted(set(show_epochs))


        
        # __________________________  LOOP OVER THE EPOCHS  ____________________________
        for epoch in range(1, epochs + 1):
            
            total_mse = 0
            total_kl = 0
            total_penalty = 0
            total_premium = 0
            train_total_loss = 0

            # _______________________________ TRAINING LOOP ___________________________________ 
            
            for data, label in self.train_loader:
                # Flatten image [Batch, 1, 28, 28] -> [Batch, 784]
                data = data.to(self.device).view(data.size(0), -1)

                if Debug: print(f"shape del tensore data(batch,785): {data.shape}")
                
                optimizer.zero_grad()
                
                # Forward Pass
                x_hat, z, b, mean, logVar = model(data, label)

                
                # Loss & Backward
                mse, kl_loss, penalty, premium, loss = self.loss_function(data, x_hat, z, b, penalize_lambda, premium_lambda, bit_type, mean, logVar, Variational)
                loss.backward()
                optimizer.step()            

                total_mse += mse.item()
                total_kl += kl_loss.item() 
                total_penalty += penalty.item()
                total_premium += premium.item()
                
                train_total_loss += loss.item() #.item() needed to extract the number in PyTorch

            if Debug: 
                print(f"epoch {epoch}")
                print(f"N={len(self.train_loader.dataset)} self.train_loader.dataset")
                print(f"length test_loader.dataset {len(self.test_loader.dataset)}")

            # _______________________________ STORAGE LOSS VALUES ___________________________________ 
            
            N = len(self.train_loader.dataset)
            avg_total = train_total_loss / N
            model.train_loss_history.append(avg_total)

            logging_string = f"[{config.name}] Epoch {epoch}/{epochs} | Train loss: {avg_total:.2f}"

            if Variational:
                avg_kl = total_kl / N
                model.kl_history.append(avg_kl)
                # logging_string += f" | KL: {avg_kl:.2f}"
            
            if penalize_lambda:
                avg_mse = total_mse / N
                avg_penalty = total_penalty / N
                model.mse_history.append(avg_mse)
                model.penalty_history.append(avg_penalty)
                logging_string += f" | MSE: {avg_mse:.2f} | Penalty: {avg_penalty:.2f}"
            
            if premium_lambda:
                avg_premium = total_premium / N
                model.premium_history.append(avg_premium)
                logging_string += f" | Premium: {avg_premium:.2f}"

            # _______________________________ Validation Loop (optional) ___________________________________
            
            if Validation:
                
                val_total_loss = 0
                model.eval()
                
                with torch.no_grad():
                    
                    for data, label in self.test_loader:
                        data = data.to(self.device).view(data.size(0), -1)
                        x_hat, z, b, mean, logVar = model(data, label)
                        _, _, _, _, loss_val = self.loss_function(data, x_hat, z, b, penalize_lambda, premium_lambda, bit_type, mean, logVar, Variational)
                        val_total_loss += loss_val.item() #.item() needed to extract the number in PyTorch

                M = len(self.test_loader.dataset)
                avg_val_total = val_total_loss / M
                model.val_loss_history.append(avg_val_total)

                logging_string += f" ||| Valid loss: {avg_val_total:.2f}"

            #_____________________________ Show losses progress _________________________________
            #if Debug:

            if epoch in show_epochs:
                print(logging_string)

            
            #_____________________________ Mut Info Calculation + Storage _________________________________
            #_______________________________ Activations registrations _________________________________

            recorder.activate_recording(True)
            VAE_info(
                model=model, 
                dataset=self.test_loader.dataset, 
                device=self.device, 
                epoch=epoch, 
                num_samples=len(self.test_loader.dataset), # Activations for all test set are saved
                mi_estimator=mi_estimator,
                mi_history=mi_history,
                RecorderActivat=recorder
            )
            recorder.activate_recording(False)

        print(f"{'='*15} EXPERIMENT COMPLETED: {config.name} {'='*15}\n")

        # _______________________________OUTPUT_______________________________
        
        return {
            "params": {**config.model_params, **config.train_params, **config.mi_params}, # create only one dict with all parameters
            "model": model,
            "optimizer": optimizer,
            "recorder": recorder,
            "mi_history": mi_history
        }
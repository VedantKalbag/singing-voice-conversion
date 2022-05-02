from model_vc_og import Generator
import torch
import torch.nn.functional as F
import time
import datetime
import pandas as pd
from torchsummary import summary
import shutil
import os
import csv
import sys
from tqdm import tqdm

"""
Useful epochs:
25000- v_loss_cd = 0.000260 t_loss_cd = 0.000057

"""

class Solver(object):

    def __init__(self, vcc_loader, valid_loader, test_loader, config):
        """Initialize configurations."""
        print(config)
        # Data loader.
        self.vcc_loader = vcc_loader
        self.validation_loader = valid_loader
        self.test_loader = test_loader
        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.load = config.load
        self.lr = config.lr

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.start_epoch = 0
        self.suffix = config.suffix

        # Build the model and tensorboard.
        self.ckpt_path = config.load_ckpt_path
        self.build_model()
        # summary(self.G.cuda(), (self.dim_neck, self.dim_emb, self.dim_pre, self.freq))

    def load_ckp(self, checkpoint_fpath):
        checkpoint = torch.load(checkpoint_fpath, map_location=self.device)
        self.G.load_state_dict(checkpoint['state_dict'])
        # self.g_optimizer.load_state_dict(checkpoint['optimizer'])
        self.G.to(self.device)
        # self.g_optimizer.to(self.device)
        return checkpoint['epoch']  

    def build_model(self):
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        print(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr)
        self.G.to(self.device)
        if self.load:
            try:
                # ckp_path = "./processed_data/trained_models/checkpoint_experiment0_step100000_validloss_3.108472446911037e-05.pth"
                self.start_epoch = self.load_ckp(self.ckpt_path)
                print(f"Loaded previous checkpoint- starting from epoch {self.start_epoch}")
            except Exception as e:
                print(e)
                pass
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    #=====================================================================================================================================#
    
    def run_validation(self):
        data_loader = self.validation_loader
        try:
                x_real, emb_org = next(data_iter)
        except:
            data_iter = iter(data_loader)
            x_real, emb_org = next(data_iter) # LOAD SPECTROGRAM AND EMBEDDINGS FROM DATASET
        x_real = x_real.to(self.device, dtype=torch.float)
        emb_org = emb_org.to(self.device, dtype=torch.float)
        with torch.no_grad():
            self.G_valid = self.G.eval()
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G_valid(x_real, emb_org, emb_org)
            g_loss_id = F.mse_loss(x_real, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)   
            
            # Code semantic loss.
            code_reconst = self.G_valid(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd

            return g_loss_id, g_loss_id_psnt, g_loss_cd

    def run_test(self):
        data_loader = self.test_loader
        try:
                x_real, emb_org = next(data_iter)
        except:
            data_iter = iter(data_loader)
            x_real, emb_org = next(data_iter) # LOAD SPECTROGRAM AND EMBEDDINGS FROM DATASET
        x_real = x_real.to(self.device, dtype=torch.float)
        emb_org = emb_org.to(self.device, dtype=torch.float)
        with torch.no_grad():
            self.G_valid = self.G.eval()
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G_valid(x_real, emb_org, emb_org)
            g_loss_id = F.mse_loss(x_real, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)   
            
            # Code semantic loss.
            code_reconst = self.G_valid(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd

            return g_loss_id, g_loss_id_psnt, g_loss_cd

    def log_values(self, path, loss, train, epoch):
        if train == 'train':
            with open(os.path.join(path,'logs', f'train_{self.suffix}.csv'),'a') as out_file: # need "a" and not w to append to a file, if not will overwrite
                writer=csv.writer(out_file, delimiter=',',lineterminator='\n',)
                row=[epoch, loss['T/loss_id'],loss['T/loss_id_psnt'], loss['T/loss_cd']]
                writer.writerow(row)
        if train == 'valid':
            with open(os.path.join(path,'logs', f'valid_{self.suffix}.csv'),'a') as out_file: # need "a" and not w to append to a file, if not will overwrite
                writer=csv.writer(out_file, delimiter=',',lineterminator='\n',)
                row=[epoch, loss['V/loss_id'],loss['V/loss_id_psnt'], loss['V/loss_cd']]
                writer.writerow(row)
    
    def log_model(self, loss, epoch):
        checkpoint = {
                        'epoch': epoch,
                        'state_dict': self.G.state_dict(),
                        'optimizer': self.g_optimizer.state_dict()
                     }
        torch.save(checkpoint, f'./processed_data/trained_models/checkpoint_experiment{self.suffix}_step{epoch}_trainloss_{loss}.pth')


        
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        # Print logs in specified order
        keys = ['T/loss_id','T/loss_id_psnt','T/loss_cd']
        try:
            # Start training.
            print('Start training...')
            start_time = time.time()
            for i in tqdm(range(self.start_epoch+1, self.num_iters), initial=self.start_epoch+1, total=self.num_iters):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch data.
                try:
                    x_real, emb_org = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, emb_org = next(data_iter) # LOAD SPECTROGRAM AND EMBEDDINGS FROM DATASET
                
                # print("Setting inputs to GPU")
                x_real = x_real.to(self.device, dtype=torch.float)
                emb_org = emb_org.to(self.device, dtype=torch.float)
                            

                # =================================================================================== #
                #                               2. Train the generator                                #
                # =================================================================================== #
                
                self.G = self.G.train()
                            
                # Identity mapping loss
                x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
                g_loss_id = F.mse_loss(x_real, x_identic)   
                g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)   
                
                # Code semantic loss.
                code_reconst = self.G(x_identic_psnt, emb_org, None)
                g_loss_cd = F.l1_loss(code_real, code_reconst)


                # Backward and optimize.
                g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss = {}
                loss['T/loss_id'] = g_loss_id.item()
                loss['T/loss_id_psnt'] = g_loss_id_psnt.item()
                loss['T/loss_cd'] = g_loss_cd.item()
                # t_iteration.append(i+1)
                # t_loss_id.append(g_loss_id.item())
                # t_loss_psnt.append(g_loss_id_psnt.item())
                # t_loss_cd.append(g_loss_cd.item())

                # =================================================================================== #
                #                                 4. Validation                                    #
                # =================================================================================== #
                if (i+1) % 100 == 0 and i != 0:
                    print("Running validation:")
                    v_keys = ['V/loss_id','V/loss_id_psnt','V/loss_cd']
                    loss_id, loss_id_psnt, loss_cd = self.run_validation()
                    v_loss={}
                    v_loss['V/loss_id'] = loss_id.item()
                    v_loss['V/loss_id_psnt'] = loss_id_psnt.item()
                    v_loss['V/loss_cd'] = loss_cd.item()
                    log = "Validation after iteration [{}/{}]".format(i+1, self.num_iters)
                    for tag in v_keys:
                        log += ", {}: {:.6f}".format(tag, v_loss[tag])
                    print(log)
                    # v_iteration.append(i+1)
                    # v_loss_id.append(loss_id.item())
                    # v_loss_psnt.append(loss_id_psnt.item())
                    # v_loss_cd.append(loss_cd.item())
                    self.log_values(f'./processed_data/trained_models', v_loss, 'valid', i+1)


                # =================================================================================== #
                #                                 5. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training information.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                    for tag in keys:
                        log += ", {}: {:.6f}".format(tag, loss[tag])
                    print(log)
                    self.log_values(f'./processed_data/trained_models', loss, 'train', i+1)

                # Log model
                if (i+1) % 100 == 0 :
                    print(i+1)
                    n = (i+1) - self.start_epoch
                    if 10000 < (n) < 100000 :
                        if (n) % 10000 == 0:
                            print('10k-100k')
                            print("Saving model checkpoint")
                            self.log_model(g_loss_cd.item(), n)
                        else:
                            continue
                    if (n) > 100000:
                        if (n) % 25000 == 0:
                            print('>100k')
                            print("Saving model checkpoint")
                            self.log_model(g_loss_cd.item(), n)
                        else:
                            continue
                    elif (n) < 10000:
                        print("<10k")
                        print("Saving model checkpoint")
                        self.log_model(g_loss_cd.item(), n)
                        
            

            print("Training done - saving model checkpoint")
            self.log_model(g_loss_cd.item(), i+1)
            t_loss_id, t_loss_id_psnt, t_loss_cd = self.run_test()
            test_loss={}
            test_loss['loss_id'] = t_loss_id
            test_loss['loss_id_psnt'] = t_loss_id_psnt
            test_loss['loss_cd'] = t_loss_cd

            log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
            for tag in ['loss_id','loss_id_psnt','loss_cd']:
                log += ", {}: {:.6f}".format(tag, test_loss[tag])
            print(log)
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt, saving model")
            self.log_model(g_loss_cd.item(), i+1)
            sys.exit()
        
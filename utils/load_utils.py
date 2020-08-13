



def load_model(arch, specs, exp_mode=''):
    latent_size = specs["CodeLength"]
    if exp_mode == "IGR" or exp_mode == "IGR_net_loss":
        decoder = arch.Decoder(3+latent_size, **specs["NetworkSpecs"]).cuda()
    else:
        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
    return decoder
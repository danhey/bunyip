    # # def initialize_network(self, model_path=None):
    # #     """ Move this to a class please"""
    # #     try:
    # #         from tensorflow.keras.models import load_model
    # #         from tensorflow.keras.initializers import glorot_uniform
    # #         from tensorflow.keras.utils import CustomObjectScope
    # #     except:
    # #         raise ImportError("You need TensorFlow for this")

    # #     if model_path is None:
    # #         import os
    # #         model_path = os.path.join(os.path.dirname(__file__),  "network/RELU_2000_2000_lr=1e-05_norm_insert2000layer-1571628331/NN.h5")
        
    # #     with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    # #         model = load_model(model_path)
    # #     return model
        
    # def predict(self, binned_flux, model=None, norm=True):
    #     """Predict light curve parameters from the neural network
        
    #     Parameters
    #     ----------
    #     binned_flux : np.ndarray
    #         Flux values
    #     model : [type], optional
    #         [description], by default None
    #     norm : bool, optional
    #         [description], by default True
        
    #     Returns
    #     -------
    #     [type]
    #         [description]
    #     """
    #     if model is None:
    #         model = self.initialize_network()
    #     y_hat = model.predict(binned_flux[None,:])
            
    #     if norm:
    #         mu= np.array([
    #             0.6205747949371053,
    #             0.2374090928468623,
    #             0.1891195153173617,
    #             1.3006089700783283,
    #             69.9643427508551,
    #             0.17749621516829056,
    #             179.6479435131075,
    #         ])
    #         std = np.array([
    #             0.22820790194476795,
    #             0.08166430725337233,
    #             0.05891981424090313,
    #             0.4059874833585892,
    #             11.465339377838976,
    #             0.12821797216376407,
    #             103.59690197983575,
    #         ])
    #         y_hat = y_hat * std + mu
    #     return y_hat


        # def update_from_network(self, **kwargs):
    #     """Update light curve parameters from neural network model.
    #     """
    #     lc = lk.LightCurve(self.phase, self.flux)
    #     binned = lc.bin(bins=100).normalize()
    #     prediction = self.predict(binned.flux, **kwargs)[0]
    #     self.update_parameters(prediction)
    
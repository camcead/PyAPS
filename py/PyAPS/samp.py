from astropy.samp import SAMPIntegratedClient



class Aladin:
    def __init__(self):
        self.client = SAMPIntegratedClient()
        self.client.connect()

    def aladin_client(self):
        clients = self.client.get_registered_clients()
        aladin_id = None
        for clid in clients:
            if not len(self.client.get_metadata(clid)):
                continue
            if self.client.get_metadata(clid)['samp.name'] == 'Aladin':
                aladin_id = clid
                break
        return aladin_id

    def aladin_point(self,ra,dec):
        clid = self.aladin_client()
        message = {}
        message["samp.mtype"] = "coord.pointAt.sky"
        params = {}
        params['ra'] = "{}".format(ra)
        params['dec'] = "{}".format(dec)
        message["samp.params"] = params
        self.client.notify(clid, message)

    def aladin_script(self,script):
        clid = self.aladin_client()
        message = {"samp.mtype":"script.aladin.send"}
        message["samp.params"] = {"script" : script}
        self.client.notify(clid, message) 

    def load_fov(self,req='IFU',prompt=True):
        import glob
        import os
        selector = {}
        selector['LIFU'] = ['weaveLIFU.vot']
        selector['mIFU'] = ['weave.vot','weavemIFU.vot']
        selector['IFU'] = ['weave.vot','weaveLIFU.vot','weavemIFU.vot']
        loc = os.path.dirname(os.path.realpath(__file__))
        fov_loc = os.path.join(loc,"../fov/")
        for _vot in selector[req]:
            fov_file = os.path.join(fov_loc,_vot)
            self.aladin_script("load {}".format(fov_file))

    def jump_to(self, ra, dec, FOV=120):
        self.aladin_script("goto J2000 {} {}".format(ra,dec))
        self.aladin_script(f"zoom {FOV}arcmin")
        
    def load_fits(self,fits_file):
        self.aladin_script("load {}".format(fits_file))



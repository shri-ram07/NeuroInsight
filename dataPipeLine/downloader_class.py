class downloader:
    def __init__(self , token):
        self.token = token

    def download(self):
        try:
            import synapseclient
            import synapseutils
        except ImportError:
            print("Please install Synapse")
            print("Run - pip install synapseclient | in terminal")

        syn = synapseclient.Synapse()
        syn.login(authToken=self.token)
        files = synapseutils.syncFromSynapse(syn, 'syn51156910')
        print("Download Complete")




SP_METHODS = ['log', 'log2', 'log10', 'sqrt']

class InvalidTransformation(Exception):

    def __init__(self, transformation_name:str, allowed_transformations:list):
        self.name = transformation_name
        self.allowed_transformations = allowed_transformations

    def __str__(self):
        return f"""
        {self.name} is not a valid transformation.
        Allowed transformations are \n{self.allowed_transformations}
        """

class TransformerNotFittedError(Exception):

    def __str__(self):
        return """
            You are probably calling inverse_transform without fitting the transfomrer
            first. Either run .fit_transform first or provide scaler which was 
            used to transform or key to fetch the scaler
            """
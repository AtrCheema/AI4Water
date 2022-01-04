

class InvalidTransformation(Exception):

    def __init__(self, transformation_name:str, allowed_transformations:list):
        self.name = transformation_name
        self.allowed_transformations = allowed_transformations

    def __str__(self):
        return f"""
        {self.name} is not a valid transformation.
        Allowed transformations are \n{self.allowed_transformations}
        """
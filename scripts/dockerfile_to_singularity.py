from spython.main.parse.writers import get_writer
from spython.main.parse.parsers import get_parser

DockerParser = get_parser('docker')
SingularityWriter = get_writer('singularity')
# from spython.main.parse.writers import SingularityWriter

parser = DockerParser('C:/Users/Ben/Documents/dotaservice/dockerfiles/Dockerfile-dota')
writer = SingularityWriter(parser.recipe)
print(writer.convert())
result = writer.convert()
with open('C:/Users/Ben/Documents/dotaservice/dockerfiles/Singularity-dota.simg', 'w') as f:
    f.write(result)

parser = DockerParser('C:/Users/Ben/Documents/dotaservice/dockerfiles/Dockerfile-dotaservice')
writer = SingularityWriter(parser.recipe)
print(writer.convert())
result = writer.convert()
with open('C:/Users/Ben/Documents/dotaservice/dockerfiles/Singularity-dotaservice.simg', 'w') as f:
    f.write(result)
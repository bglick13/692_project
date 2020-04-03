from spython.main.parse.writers import get_writer
from spython.main.parse.parsers import get_parser

DockerParser = get_parser('docker')
SingularityWriter = get_writer('singularity')
# from spython.main.parse.writers import SingularityWriter

parser = DockerParser('/Users/benjaminglickenhaus/PycharmProjects/dotaservice/dockerfiles/Dockerfile-dota')
writer = SingularityWriter(parser.recipe)
print(writer.convert())
result = writer.convert()
with open('/Users/benjaminglickenhaus/PycharmProjects/dotaservice/dockerfiles/Singularity-dota.simg', 'w') as f:
    f.write(result)

parser = DockerParser('/Users/benjaminglickenhaus/PycharmProjects/dotaservice/dockerfiles/Dockerfile-dotaservice')
writer = SingularityWriter(parser.recipe)
print(writer.convert())
result = writer.convert()
with open('/Users/benjaminglickenhaus/PycharmProjects/dotaservice/dockerfiles/Singularity-dotaservice.simg', 'w') as f:
    f.write(result)
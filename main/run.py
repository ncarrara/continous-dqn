import cross_comparaison
import generate_samples
import learn_autoencoders
import test_and_base
import transfer_dqn
import no_transfer_dqn
from configuration import C

# c.C.load("config/0.json")
# generate_samples.main()
# learn_autoencoders.main()
# cross_comparaison.main()
# test_and_base.main()


C.load("config/0.json")
print(C)
# transfer_dqn.main()
no_transfer_dqn.main()


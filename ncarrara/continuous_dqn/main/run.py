from ncarrara.continuous_dqn.main import clean, generate_samples, learn_autoencoders, test_and_base, transfer_dqn
from ncarrara.continuous_dqn.tools.configuration import C


C.load("config/0_random.json")
# C.load("config/0_pydial.json")

clean.main()
generate_samples.main()
learn_autoencoders.main()
test_and_base.main()
# no_transfer_dqn.main()
transfer_dqn.main()
transfer_dqn.show()




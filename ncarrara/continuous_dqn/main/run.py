from ncarrara.continuous_dqn.main import generate_samples, learn_autoencoders, test_and_base, transfer_dqn, \
    no_transfer_dqn
from ncarrara.continuous_dqn.tools.configuration import C



# C.load("config/0_random.json")#.create_fresh_workspace()
# C.load("config/0_slot_filling.json").create_fresh_workspace()
C.load("config/0_pydial.json")#.create_fresh_workspace()

# generate_samples.main()
# learn_autoencoders.main()
# test_and_base.main()
# no_transfer_dqn.main()
# transfer_dqn.main()
transfer_dqn.show()




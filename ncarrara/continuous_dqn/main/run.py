from ncarrara.continuous_dqn.main import generate_samples, learn_autoencoders, test_and_base, transfer_dqn
from ncarrara.continuous_dqn.tools.configuration import C



C.load_pytorch().load("config/tests/main.json")#.create_fresh_workspace(force=True)
# C.load_pytorch().load("config/0_random.json")
# C.load_pytorch().load("config/0_slot_filling.json")
# C.load("config/0_pydial.json").load_pytorch()#.create_fresh_workspace()

# generate_samples.main()
# learn_autoencoders.main()
# test_and_base.main()
transfer_dqn.main()
# transfer_dqn.show()




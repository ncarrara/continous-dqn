from continuous_dqn.tools.configuration import C
import continuous_dqn.main.transfer_dqn as transfer_dqn

C.load("config/0_random.json")
# C.load("config/0.json")
# C.load("config/test_mountain_car.json")
# C.load("config/lunar_lander_0.json")

# WARNINNNNNNG #
# clean.main()
# generate_samples.main()
# learn_autoencoders.main()
# test_and_base.main()
# no_transfer_dqn.main()
transfer_dqn.main()
# transfer_dqn.show()




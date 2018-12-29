from continuous_dqn.tools.configuration import C
import continuous_dqn.main.transfer_dqn as transfer_dqn
import continuous_dqn.main.learn_autoencoders as learn_autoencoders
import continuous_dqn.main.generate_samples as generate_samples
import continuous_dqn.main.test_and_base as test_and_base
import continuous_dqn.main.clean as clean
import continuous_dqn.main.no_transfer_dqn as no_transfer_dqn

C.load("config/0_random.json")

clean.main()

generate_samples.main()
learn_autoencoders.main()
test_and_base.main()
no_transfer_dqn.main()
transfer_dqn.main()
transfer_dqn.show()




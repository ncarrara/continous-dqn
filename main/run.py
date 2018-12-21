import cross_comparaison
import generate_samples
import learn_autoencoders
import test_and_base
import configuration as c

c.C.load("config/0.json")
generate_samples.main()
learn_autoencoders.main()
cross_comparaison.main()
test_and_base.main()
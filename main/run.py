import cross_comparaison
import generate_samples
import learn_autoencoders
import configuration as c

c.C.load("config/0.json")
generate_samples.main()
learn_autoencoders.main()
cross_comparaison.main()

c.C.load("config/1.json")
generate_samples.main()
learn_autoencoders.main()
cross_comparaison.main()

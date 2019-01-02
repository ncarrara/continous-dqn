from ncarrara.bftq_pydial.main import compare_ftq_vs_bftq, create_data
from ncarrara.bftq_pydial.tools.configuration import C

# C.load("config_main_pydial/test2.json")#.create_fresh_workspace()
# C.load("config_main_pydial/test3.json")#.create_fresh_workspace()
C.load("config_main_pydial/test4.json")#.create_fresh_workspace()

create_data.main()
for lambda_ in [0,50,100,1000]:
    compare_ftq_vs_bftq.main(lambda_=lambda_)
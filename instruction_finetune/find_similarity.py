from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

embedding_doc1 = model.encode("Channeling money is the process of transferring money or assets from a person\'s account to the financial system, "
                              "typically through various means such as loans, investments, and dividends. The process involves the transfer of "
                              "money from individuals' savings accounts to the financial system, either through the transfer of money from their "
                              "account to a bank or a mutual fund, or by the transfer of funds from their accounts to a company or government.")

embedding_doc2 = model.encode("As outlined, the financial system consists of the flows of capital that take place between individuals and households (personal finance), governments (public finance), and businesses (corporate finance). Finance thus studies the process of channeling money from savers and investors to entities that need it. Savers and investors have money available which could earn interest or dividends if put to productive use. Individuals, companies and governments must obtain money from some external source, such as loans or credit, when they lack sufficient funds to run their operations.")
similarity = model.similarity(embedding_doc1, embedding_doc2)
print(similarity)

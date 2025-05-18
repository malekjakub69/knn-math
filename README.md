# KNN-Math

Tento projekt slouží k rozpoznávání matematických výrazů z obrázků a jejich převodu do LaTeX zápisu. Pro strojové učení využíváme knihovnu **PyTorch**.

## Dokumentace

Podrobná dokumentace je dostupná ve složce `dokumentace`. Je psána v LaTeXu a zdrojové `.tex` soubory jsou k dispozici.

## Příprava datasetu

Ve složce `app/neural_network` se nachází hlavní soubory, které definují přípravu datasetu, HME recognition model a trénovací skript. Přípravu datasestu lze sputit příkazem:

```bash
python -m app.neural_network.prepare_dataset --input .\dataset\train_set\ --output dataset_path/
```

Trénování lze spustit pomocí:

```bash
python -m app.neural_network.train_nn --device cuda --data_dir dataset_path/ --epochs 30
```

## Relevantní zdroje a soutěže

-   [CROHME 2023](https://crohme2023.ltu-ai.dev/) - Competition on Recognition of Online Handwritten Mathematical Expressions
-   [ICDAR](https://ai.100tal.com/icdar) - International Conference on Document Analysis and Recognition
-   [Research paper 1](https://link.springer.com/chapter/10.1007/978-3-031-41679-8_33)
-   [Research paper 2](https://link.springer.com/chapter/10.1007/978-3-031-41679-8_34)

# OCR spécifique pour Mon Devis Sans Oublis

Afin de traiter au mieux les images déposées via OCR différentes briques sont mises à contribution et encore en évaluation :
* pour le découpage des images en pages :
  * `ffmpeg`
* pour la reconnaissance des images et lire leur contenu via OCR
  * `surya-ocr` (Python) 
  * `tesseract` (natif)
* pour transformer les PDF en images
  * librairie Poppler `pdftoppm` (natif)
  * la gem MiniMagick (IM) `mini_magick` avec ImageMagick 6.9 (comme sur Scalingo) (natif)

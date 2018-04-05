## Оцифрованный список задач <sup>TM</sup>

* Administrative
  * Github
  * Этот листок
  * Telegram + 2 человека
  * Время (Для согласования пробуем использовать [дудл](https://doodle.com/poll/kkb3ed7aiex2su59))
  * GPU

* Baseline
  * Данные
    * WMT
    * OPUS
    * [JRC-Arquis](https://wt-public.emm4u.eu/Acquis/JRC-Acquis.3.0/corpus/)
      * он не супербольшой (~750k пар)
      * его используют авторы статьи
      * [en-de.zip](http://opus.nlpl.eu/download.php?f=JRC-Acquis/de-en.txt.zip)
      * [en-fr.zip](http://opus.nlpl.eu/download.php?f=JRC-Acquis/en-fr.txt.zip)
      * [en-es.zip](http://opus.nlpl.eu/download.php?f=JRC-Acquis/en-es.txt.zip)
    * **Транслит**
    * Restricted domain
    * Low resource
  * Режим обучения
      * малое число нейронов
      * половина корпуса
  * Библиотека
    * t2t
    * **свое**
        
        Новые идеи от 05.03:

        * Скрипт обучения (батчсайз, lr, количество нейронов - 128?)
        * Скрипт валидации
        * Стандартизировать словарь (маппинг из слов в id)
    * nematus
      [Нет, т.к theano]
    * opennmt ?
  * Поиск
  * Выбрать и запустить, перевести
  * Транслит
    * RTL
    * multiple refs
    * dev/train/test/test'
    * Разнородные 600К/1К
    
    Новые идеи от 05.03:

    * Идея: преобразовать char-by-char for sanity check
    * Какую задачу решаем: огласовку или транслита сразу на английский
    * Провалидировать ambiguity
    * Канонический сплит 50/50 train
    * Взять маленький кусочек из train'а для того, чтобы на нем делать лучше sanity check'и в месте с тестом
  
  * Данные
  * Поисковый движок для транслита
    * Написать
* Статьи
  * [Главная статья](https://arxiv.org/pdf/1705.07267.pdf)
  * Дима - [Improving Neural Machine Translation through Phrase-based Forced Decoding](https://arxiv.org/pdf/1711.00309.pdf)
  * Паша - [Incorporating Discrete Translation Lexicons into Neural Machine Translation](https://arxiv.org/pdf/1606.02006.pdf)
  * Федя - [Two are Better than One: An Ensemble of Retrieval- and Generation-Based Dialog Systems](https://arxiv.org/pdf/1610.07149.pdf)
  * Степа - [Generating Sentences by Editing Prototypes — text generation](https://arxiv.org/pdf/1709.08878.pdf)


-------
Итого: задачи на следующий раз (после 05.03)

* Обработать данные
* Написать бейзлайн (со всеми скриптами)
* Дочитать статьи и определиться с начальными изменениями (с конкретикой)

-----
Новые идеи подъехали:

* [Настроить таки детерминированные пайплайны. Не супер новое, но важное]
* Сделать возможность делать diff на разных моделях
* Have a way to check hyperparameters

* Look at the errors baseline make. Retrain the baseline with less data. (Maybe). Recomute baseline with 1, 1/2, 1/4, 1/8, ...? What errors does each model make? Hopefully, more => better. Try and understand what kind of errors we make. Be ready to explain what we've found.
* Read at least one paper and be prepared to explain problems. Look for important things they missed out.
Chose a paper based by Monday (evening?)
* Maybe oracle?
* Implement very basic main-article-like approach

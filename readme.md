# NeuroWorkshop
## Интенсив по нейронным сетям и глубокому обучению

Данный интенсив предназначен как первоначальное введение в тематику нейронных сетей для разработчиков, желающих применять соответствующие технлологии в своих проектах. Интенсив проводился в рамках [большой июньской школы DevCon 2017](http://events.techdays.ru/Future-Technologies/2017-06/).
Видеозапись интенсива доступна [на нашем канале YouTube](https://www.youtube.com/watch?v=9haeWybwCNk&list=PLVDsxiCH_PqRIZ84g-1X57Tr6VHBVVevP), а
презентации - [здесь](https://github.com/evangelism/DevCon-School/tree/master/Big%20June%20School/Intensives/NeuroWorkshop). 

Для общения - присоединяйтесь к чату [@neuroworkshop](http://telegram.me/neuroworkshop)

Первым делом - [зайдите сюда](https://notebooks.azure.com/sosh/libraries/neuroworkshop?WT.mc_id=academic-0000-dmitryso)

## О чем будет рассказано на интенсиве

  * Microsoft и искусственный интеллект - стратегия
  * [Введение в нейросети - персептроны](Notebooks/Perceptrons.ipynb)
  * [Многослойные сети прямого распространения на CNTK](Notebooks/IntroCNTK.ipynb)
      - [Лабораторная работа: Iris Dataset](Notebooks/Lab_Iris.ipynb)
      - [Лабораторная работа: MNIST Dataset](Notebooks/Lab_MNIST.ipynb)
  * [Свёрточные сети для анализа изображений](Notebooks/ConvolutionalNets.ipynb)
      - [Лабораторная работа: MNIST Convolution](Notebooks/Lab_MNIST.ipynb)
  * Различные приёмы обучения нейросетей
  * Рекуррентные нейронные сети для анализа последовательностей
      - [Лабораторная работа: LSTM и TensorFlow](Notebooks/LSTM.ipynb)
  * Финальные соревнования: [кошки против собак](Notebooks/Cats_Dogs.ipynb)
  * Обучение нейросетей в промышленном масштабе

## Первые шаги

Для проведения интенсива вам потребуется:
  * **Возможность работать с [Azure Notebooks](https://notebooks.azure.com/?WT.mc_id=academic-0000-dmitryso)**, т.е. компьютер с браузером и Microsoft Account. Создать Microsoft Account можно на http://outlook.com, заведя там почтовый ящик.
  * **Виртуальная машина с GPU в облаке Microsoft Azure** (для выполнения финального задания). Для этого нужна подписка на Microsoft Azure.

### Делаем копию всех материалов в Azure Notebooks

 1. Заходим по адресу https://notebooks.azure.com/sosh/libraries/neuroworkshop?WT.mc_id=academic-0000-dmitryso
 2. Нажимаем **Clone and Run**
 3. При необходимости заходим с Microsoft Account.

С помощью Azure Notebooks можно будет выполнить большинство заданий интенсива, но для выполенения финального задания, и для быстрого экспериментирования, будет удобнее использовать виртуальную машину с GPU.

### Создаём виртуальную машину с GPU

Для начала, вам нужна облачная подписка. Если у Вас её нет, получить её можно так:

  1. На мероприятии, если вам раздали код Azure Pass - следуем инструкции http://aka.ms/azpass
  2. Если вы смотрите онлайн или решили пройти мастер-класс самостоятельно - получаем trial-подписку на месяц 
     [по ссылке](https://azure.microsoft.com/free/?WT.mc_id=academic-0000-dmitryso). Для этого потребуется кредитная карта.

В обоих случаях **надо использовать Microsoft Account, который не был привязан ранее к облачной подписке**. В противном случае вы не сможете привязать к такому аккаунту пробную подписку.

Процесс создания виртуальной машины с GPU подробно описан тут: http://bit.ly/datasciencevm

К сожалению, на подписках Azure Trials и Azure Pass иногда бывает сложно создать виртуальную машину с GPU, поскольку ресурсов GPU в облаке в целом не хватает. Поэтому рекомендуем вам попробовать (в следующем порядке):

 1. Создать виртуальную машину Data Science Virtual Machine - Windows 2016 типа N-Series NC6 (в одном из доступных регионов: South-Central US, East US, North Europe - но можно попробовать и другие)
 2. Если не получилось - создать такую же машину на базе N-Series NV6 (таких машин значительно больше, поэтому вероятность успеха выше)
 3. Попробовать воспользоваться одной из заранее созданных машин - информация будет на мероприятии
 4. Использовать машину с CPU, выбрав CPU помощнее

### Конфигурируем машину для работы на интенсиве

 1. Войдите в созданную машину при помощи удалённого рабочего стола
 2. Откройте консоль `cmd.exe`
 3. Перейдите в директорию `c:\dsvm` - `cd \dsvm`
 4. Склонируйте этот репозиторий: `git clone http://github.com/shwars/NeuroWorkshop`
 5. Перейдите в директорию `cd NeuroWorkshop\Utils`
 6. Запустите `setup.bat`. Этот файл сделает следующее:
      - скопирует все необходимые Azure Notebooks в папку `\dsvm\notebooks\NeuroWorkshop`, чтобы они стали вам доступны
      - подключит удалённый диск с данными для обучения как диск `L:`
      - скопирует обучающие данные для задачки *кошки против собак* на локальный диск в папку `C:\Cats_Dogs`
 7. После этого вы можете:
      - работать в удалённом доступе, открыв браузер на адрес `https://localhost:9999`
      - закрыть удалённый доступ, и обратиться к удалённой машине через браузер `http://<address>:9999`

Если вы не можете войти в Azure Notebook по указанному адресу, убедитесь, что:
      - вы настроили пароль и автозапуск Azure Notebooks, как описано [тут](http://bit.ly/datasciencevm)
      - вы настроили доступ к удалённой машине по порту 9999

### Работа на лабораторных машинах

Если вам не удалось создать в облаке свою виртуальную машину, вы можете воспользоваться одной из доступных лабораторных машин:

  * https://smart1.southcentralus.cloudapp.azure.com:9999/?WT.mc_id=academic-0000-dmitryso
  * https://smart2.eastus.cloudapp.azure.com:9999/?WT.mc_id=academic-0000-dmitryso
  * https://smart8.southcentralus.cloudapp.azure.com:9999/?WT.mc_id=academic-0000-dmitryso

Пароль и инструкции будут предоставлены на занятии.

## Полезные материалы 

 * [Стандартные примеры по CNTK в Azure Notebooks](https://notebooks.azure.com/cntk/libraries/tutorials?WT.mc_id=academic-0000-dmitryso)
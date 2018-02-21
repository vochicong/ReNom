ReNom TAG
========

Introduction to ReNom TAG
-------------------------

Tagging tool for object detection.

**How to start**

1.Move to ReNomTAG directory using following command.

``cd ReNomTAG``

2.Run server.py script and the application server starts.

``python server.py``

If the server starts, you will see a message like below.

   .. figure:: ../images/other/server_run.png
      :scale: 50%

**How to use**

The following videos describes how to use ReNomTAG. In this video, the
Oxford-IIIT Pet Dataset is used.

-  | Cats and Dogs Classification
   | https://github.com/JDonini/Cats\_Dogs\_Classification

-  | O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar Cats and Dogs
   | IEEE Conference on Computer Vision and Pattern Recognition, 2012
     Bibtex
   | http://www.robots.ox.ac.uk/~vgg/data/pets/

1.Set image data to RaNomTAG.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As following video showing, please put the image you want to tag in the
``ReNomTAG/dataset`` folder.

\*The introduction movie will be uploaded later. Currently the video is
in the folder ``static/movie/movie01.mp4``.

2.Run ReNomTAG
^^^^^^^^^^^^^^

Same as before mentioned in 'How to start ReNom', following video
describes how to start ReNomTAG.

\*The introduction movie will be uploaded later. Currently the video is
in the folder ``static/movie/movie02.mp4``.

3.How to use ReNomTAG
^^^^^^^^^^^^^^^^^^^^^

It is a video tagging by ReNom tag.

\*The introduction movie will be uploaded later. Currently the video is
in the folder ``static/movie/movie03.mp4``.

4.Recognize generated xml file.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Xml files are generated into the label directory.

\*The introduction movie will be uploaded later. Currently the video is
in the folder ``static/movie/movie04.mp4``.

**Shortcut Keys**

You can use shortcut keys to select the bounding box' class name.

In the following class name list, shortcut key 'q' is set to the class
name 'dog'. Similarly, the key 'w' is set to the class name 'cat'.

   .. figure:: ../images/other/class_list.png
      :scale: 50%

You can use these shortcuts when you select a bounding box. You can
assign the corresponding class name by pressing the shortcut key while
selecting the box.

When you want to save the xml file, you can use the 'space' key as a
saving button. The 'space' key is a shortcut of the button 'Save >>'.

   .. figure:: ../images/other/save_button.png
      :scale: 50%

**Notification**

ReNomTAG only accepts alphanumeric characters. Thus you can't input non
alphanumeric characters to class name list.

**Format of xml file**

The format of the xml file which created by ReNomTAG follows `PASCAL
VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`__ format.

An example is bellow.

::

    <annotation>
     <folder>
      dataset
     </folder>
     <filename>
      2007_000027.jpg
     </filename>
     <object>
      <pose>
       Unspecified
      </pose>
      <name>
       cat
      </name>
      <truncated>
       0
      </truncated>
      <difficult>
       0
      </difficult>
      <bndbox>
       <ymax>
        203.02013422818794
       </ymax>
       <xmin>
        134.7902328154634
       </xmin>
       <xmax>
        238.81923552543284
       </xmax>
       <ymin>
        104.02684563758389
       </ymin>
      </bndbox>
     </object>
     <source>
      <database>
       Unknown
      </database>
     </source>
     <path>
      dataset/2007_000027.jpg
     </path>
     <segments>
      0
     </segments>
     <size>
      <width>
       486
      </width>
      <height>
       500
      </height>
      <depth>
       3
      </depth>
     </size>
    </annotation>

Installation
------------

For installing ReNomTAG, download this repository.

``git clone https://github.com/ReNom-dev-team/ReNomTAG``

And move into ReNomTAG directory.
``cd ReNomTAG``

Then install all required packages.

``pip install -r requirements.txt``


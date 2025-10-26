NGDC Marine Trackline Geophysics Web Retrieval readme.txt            3 June 2014

This describes data downloaded using the on-line map.


***************************
File/Directory Descriptions
***************************
After uncompressing, downloaded data files will be in a folder named /MGD77_######/.

Files:
    mgd77.pdf - MGD77/MGD77T format description
    microfilm_readme.txt - information about scanned microfilm files
    readme.txt - this readme file

MGD77 Directory:
  /MGD77_######/
    Possible files are:
    [Survey_ID, NGDC_ID, or MGD77_######].m77t - tab-delimited MGD77T data
    [Survey_ID, NGDC_ID, or MGD77_######].h77t - tab-delimited MGD77T metadata
    [Survey_ID, NGDC_ID, or MGD77_######].xyz - tab-delimited xyz data (longitude,latitude,value)
        Units of measure in XYZ are the same as defined for MGD77T
        (e.g. longitude and latitude are in decimal degrees, depths are in meters)
        with the exception that depths in XYZ are negative values. 

Additional Data Directories:
  /[Data_Type]/[Survey_ID]/
    Data files or further data designation


***************************
More Information
***************************
For additional data types and services see:
http://www.ngdc.noaa.gov/mgg/mggd.html

NGDC's GEODAS-NG (GEOphysical DAta System - Next Generation) desktop software
tools can be used for working with data downloaded from NGDC's trackline web
presence, especially NGDC-developed downloaded formats. While this software
suite is expected to work well for some time, support in the form of updates
and other changes may not be available after late 2014.  It may be useful
to explore other commercial and open-source tools to determine if they meet
your needs.

For NGDC GEODAS-NG software see:
http://www.ngdc.noaa.gov/mgg/geodas/geodas.html

For questions or comments contact:
trackline.info@noaa.gov

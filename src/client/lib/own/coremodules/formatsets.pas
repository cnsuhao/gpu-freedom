unit formatsets;
{
 TFormatSet is used to override system specific international settings,
 so that GPU across continents can communicate time stamps, floating
 point numbers and dates. The German standard is used.

}
interface

uses SysUtils;

implementation


initialization

  DecimalSeparator := '.';
  TimeSeparator    := ':';
  DateSeparator    := '.';

end.



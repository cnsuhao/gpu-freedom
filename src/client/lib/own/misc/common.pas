unit common;
{
 This unit defines system wide setting, like the core version and a short
 description of the current compiled version.
 
 Constants for communication with frontends are defined here.
 Directories are defined here, too.
 
 TFormatSet is used to override system specific international settings,
 so that GPU across continents can communicate time stamps, floating
 point numbers and dates. The German standard is used.

}
interface

uses SysUtils;

var GPU_Version:String = '0.963';

{to make GPU independent of OS format settings in
 control panel - international settings}

type
  TCustomFormatSet = class
    DecimalSeparator : String;
    TimeSeparator    : String;
    DateSeparator    : String;
  end;

type TFormatSet = class(TObject)
   public
     fs :
     {$IFDEF D7}
     TFormatSettings
     {$ELSE}
     TCustomFormatSet
     {$ENDIF}
     ;
     constructor Create;
end;


implementation

constructor TFormatSet.Create;
begin
  inherited Create;
  {$IFNDEF D7}
  fs := TCustomFormatSet.Create;
  {$ELSE}
  GetLocaleFormatSettings(1031, fs);  {1031 German_standard}
  {$ENDIF}
  fs.DecimalSeparator := '.';
  fs.TimeSeparator    := ':';
  fs.DateSeparator    := '.';
end;

end.



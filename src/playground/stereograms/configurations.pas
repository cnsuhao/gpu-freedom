unit configurations;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

type TConfiguration = record
    mEyeDist,
    mResolution,
    mZScale,
    mMu       : Extended;
end;

var conf : TConfiguration;

procedure initConfiguration();

implementation

procedure initConfiguration();
begin
  conf.mEyeDist := 2.56;   // (6.5cm in inches)
  conf.mResolution := 94; // pixels per inch
  conf.mZScale := 1/256;
  conf.mMu := 1/3;
end;

end.


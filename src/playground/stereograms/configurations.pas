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
  conf.mEyeDist := 6.5;
  conf.mResolution := 72; // 72 dpi
  conf.mZScale := 1/255;
  conf.mMu := 1/3;
end;

end.


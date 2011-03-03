unit mainapp;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  openglspherecontrol;

type
  TGPUMainApp = class(TForm)
  private
    { private declarations }
  public
    { public declarations }
  end; 

var
  GPUMainApp: TGPUMainApp;

implementation

initialization
  {$I mainapp.lrs}

end.


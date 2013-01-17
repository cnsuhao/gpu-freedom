unit stereogramform;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, ExtCtrls,
  StdCtrls, stereogramsunit, configurations;

type

  { TfrmStereogram }

  TfrmStereogram = class(TForm)
    btnLoadImage: TButton;
    btnGenerateStereogram: TButton;
    edtMonitorLength: TEdit;
    edtMonitorWidth: TEdit;
    edtResolution: TEdit;
    edtPaperDistance: TEdit;
    edtEyeDistance: TEdit;
    Label1: TLabel;
    MonitorLength: TLabel;
    lblMonitorWidth: TLabel;
    lblPaperDistance: TLabel;
    lblEyeDistance: TLabel;
    OpenDialog: TOpenDialog;
    zimg: TImage;
    stereoimg: TImage;
    procedure btnGenerateStereogramClick(Sender: TObject);
    procedure btnLoadImageClick(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
  end;

var
  frmStereogram: TfrmStereogram;

implementation

{$R *.lfm}

{ TfrmStereogram }

procedure TfrmStereogram.btnLoadImageClick(Sender: TObject);
begin
    openDialog := TOpenDialog.Create(self);
    openDialog.InitialDir := GetCurrentDir;
    openDialog.Options := [ofFileMustExist];
    openDialog.Filter :=
      'All images|*.png;*.bmp;*.gif;*.jpg';
    openDialog.FilterIndex := 2;
    if openDialog.Execute
    then
    begin
        zimg.Picture.LoadFromFile(openDialog.FileName);
        stereoimg.Picture.LoadFromFile(openDialog.FileName);
    end;

    initConfiguration;
    openDialog.Free;
end;

procedure TfrmStereogram.btnGenerateStereogramClick(Sender: TObject);
var y : Longint;
    sameArr, pDepth : TDepthDataType;
begin
    if FileExists('samearr.txt') then DeleteFile('samearr.txt');

    for y:=0 to zimg.Height-1 do
         begin
           prepareDepthArray(zimg, sameArr, y);
           makeSameArray(sameArr, pDepth, zimg.Width, 1);
           printSameArray(sameArr, zimg.Width);
           colorImageLineBlackWhite(sameArr, stereoimg, y);
         end;

     ShowMessage('Stereogram generated');
end;

end.


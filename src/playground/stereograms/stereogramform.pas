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
var x, y : Longint;
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

    for y:=0 to stereoimg.Height-1 do
      for x:=0 to stereoimg.Width-1 do
          stereoimg.Picture.Bitmap.Canvas.Pixels[x,y] := clBlack;

    showMessage('Image with is: '+IntToStr(zimg.Width));
end;

procedure TfrmStereogram.btnGenerateStereogramClick(Sender: TObject);
var y : Longint;
    sameArr, pDepth : TDepthDataType;
begin
    if FileExists('samearr.txt') then DeleteFile('samearr.txt');
    if FileExists('error.txt') then DeleteFile('error.txt');

    for y:=0 to zimg.Height-1 do
         begin
           prepareDepthArray(zimg, sameArr, y);
           makeSameArray(sameArr, pDepth, zimg.Width, 1);
           //printSameArray(sameArr, zimg.Width);
           checkSameArray(sameArr, zimg.Width, y);
           colorImageLineBlackWhite(sameArr, zimg, stereoimg, y);
         end;

     ShowMessage('Stereogram generated');
end;

end.


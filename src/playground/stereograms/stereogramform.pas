unit stereogramform;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, ExtCtrls,
  StdCtrls, stereogramsunit, configurations, isaac;

type

  { TfrmStereogram }

  TfrmStereogram = class(TForm)
    btnLoadImage: TButton;
    btnGenerateStereogram: TButton;
    btnLoadTexture: TButton;
    btnGenerateRandomTexture: TButton;
    edtMonitorLength: TEdit;
    edtMonitorWidth: TEdit;
    edtResolution: TEdit;
    edtPaperDistance: TEdit;
    edtEyeDistance: TEdit;
    texture: TImage;
    Label1: TLabel;
    MonitorLength: TLabel;
    lblMonitorWidth: TLabel;
    lblPaperDistance: TLabel;
    lblEyeDistance: TLabel;
    OpenDialog: TOpenDialog;
    zimg: TImage;
    stereoimg: TImage;
    procedure btnGenerateRandomTextureClick(Sender: TObject);
    procedure btnGenerateStereogramClick(Sender: TObject);
    procedure btnLoadImageClick(Sender: TObject);
    procedure btnLoadTextureClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
    isaac : TIsaac;
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

    texture.Picture.LoadFromFile('textures\SC_damask24.png');

    //showMessage('Image with is: '+IntToStr(zimg.Width));
end;

procedure TfrmStereogram.btnLoadTextureClick(Sender: TObject);
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
            texture.Picture.LoadFromFile(openDialog.FileName);
        end;
end;

procedure TfrmStereogram.FormCreate(Sender: TObject);
begin
  isaac := TIsaac.Create;
end;

procedure TfrmStereogram.btnGenerateStereogramClick(Sender: TObject);
var y : Longint;
    sameArr, pDepth : TDepthDataType;
begin
    if FileExists('log.txt') then DeleteFile('log.txt');
    if FileExists('samearr.txt') then DeleteFile('samearr.txt');
    if FileExists('error.txt') then DeleteFile('error.txt');

    for y:=0 to zimg.Height-1 do
         begin
           prepareDepthArray(zimg, pDepth, y);
           makeSameArray(sameArr, pDepth, zimg.Width, 1);
           //printSameArray(sameArr, zimg.Width);
           //checkSameArray(sameArr, zimg.Width, y);
           colorImageLine(sameArr, zimg, stereoimg, texture, y);
         end;

    stereoimg.Picture.saveToFile('stereogram.png');
    //ShowMessage('Stereogram generated');
end;

procedure TfrmStereogram.btnGenerateRandomTextureClick(Sender: TObject);
var x, y : Longint;
    c    : TColor;
begin
  for y:=0 to texture.Height-1 do
   for x:=0 to texture.Width-1 do
       begin
          if (isaac.Val/high(Cardinal))>0.5 then
                c := clBlack
               else
                c := clWhite;

          texture.Picture.Bitmap.Canvas.Pixels[x,y] := c;
       end;
  texture.Picture.SaveToFile('random.png');
end;

end.


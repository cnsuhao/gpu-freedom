unit stereogramform;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, ExtCtrls,
  StdCtrls;

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
    then ShowMessage('File : '+openDialog.FileName)
    else ShowMessage('Open file was cancelled');
    openDialog.Free;
end;

end.


unit nndatasetform;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls;

type

  { TNNdataSetForm }

  TNNdataSetForm = class(TForm)
    btnSelectInputFileName: TButton;
    btnSelectOutputFileName: TButton;
    edtInputFilename: TEdit;
    edtOutputFilename: TEdit;
    lblInputFilename: TLabel;
    lblInputFilename1: TLabel;
  private
    { private declarations }
  public
    { public declarations }
  end;

var
  NNdataSetForm: TNNdataSetForm;

implementation

{$R *.lfm}

end.


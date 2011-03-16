unit parametersforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  StdCtrls, ExtCtrls;

type

  { TParametersForm }

  TParametersForm = class(TForm)
    btnSave: TButton;
    btnReset: TButton;
    gbNode: TGroupBox;
    gbConnection: TGroupBox;
    gbUser: TGroupBox;
    gbConfiguration: TGroupBox;
    edtNodename: TLabeledEdit;
    edtProxyHost: TLabeledEdit;
    edtProxyPort: TLabeledEdit;
  private
    { private declarations }
  public
    { public declarations }
  end; 

var
  ParametersForm: TParametersForm;

implementation

initialization
  {$I parametersforms.lrs}

end.


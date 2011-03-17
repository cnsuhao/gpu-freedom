unit parametersforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  StdCtrls, ExtCtrls, MaskEdit, Spin;

type

  { TParametersForm }

  TParametersForm = class(TForm)
    btnSave: TButton;
    btnReset: TButton;
    cbLatDir: TComboBox;
    cbTeam: TComboBox;
    cbLonDir: TComboBox;
    cbRunOnlyIdle: TCheckBox;
    edtLonSeconds: TEdit;
    edtLonMinutes: TEdit;
    edtCity: TEdit;
    edtLatMinutes: TEdit;
    edtLatSeconds: TEdit;
    gbNode: TGroupBox;
    gbConnection: TGroupBox;
    gbUser: TGroupBox;
    gbConfiguration: TGroupBox;
    edtNodename: TLabeledEdit;
    edtProxyHost: TLabeledEdit;
    edtProxyPort: TLabeledEdit;
    edtRegion: TLabeledEdit;
    edtStreet: TLabeledEdit;
    edtZip: TLabeledEdit;
    edtDescription: TLabeledEdit;
    edtLonDegrees: TLabeledEdit;
    edtLatDegrees: TLabeledEdit;
    edtUsername: TLabeledEdit;
    edtEmail: TLabeledEdit;
    edtRealname: TLabeledEdit;
    edtHomepage: TLabeledEdit;
    lblMaxUploads: TLabel;
    lblMaxDownloads: TLabel;
    lblMaxServices: TLabel;
    lblMaxComputations: TLabel;
    lblPassword: TLabel;
    lblTeam: TLabel;
    edtPassword: TMaskEdit;
    edtMaxComputations: TSpinEdit;
    edtMaxServices: TSpinEdit;
    edtMaxDownloads: TSpinEdit;
    edtMaxUploads: TSpinEdit;
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


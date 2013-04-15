unit parametersforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  StdCtrls, ExtCtrls, MaskEdit, Spin,
  coreobjects, identities;

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
    procedure btnResetClick(Sender: TObject);
    procedure btnSaveClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
    procedure displayConfiguration;
    function  validateConfiguration : Boolean;
    procedure writeConfiguration;
  public
    { public declarations }
  end; 

var
  ParametersForm: TParametersForm;

implementation

{ TParametersForm }

procedure TParametersForm.btnResetClick(Sender: TObject);
begin
  conf.loadConfiguration();
  displayConfiguration;
end;

procedure TParametersForm.btnSaveClick(Sender: TObject);
begin
 if validateConfiguration then
    begin
     writeConfiguration;
     conf.saveConfiguration();
    end;
end;

procedure TParametersForm.FormCreate(Sender: TObject);
begin
 displayConfiguration;
end;

procedure TParametersForm.displayConfiguration;
begin
  // 1. Node
  edtNodename.Text := myGPUId.Nodename;
  edtRegion.Text := myGPUId.Region;
  edtStreet.Text := myGPUID.Street;
  edtZip.Text := myGPUID.Zip;
  edtCity.Text := myGPUId.City;
  edtDescription.Text := myGPUID.Description;

  cbTeam.Text := myGPUId.Team;

  //TODO: make it work for all adjacent components
  edtLonDegrees.Text := FloatToStr(myGPUId.Longitude);
  edtLatDegrees.Text := FloatToStr(myGPUId.Latitude);

  // 2. Connection
  edtProxyHost.Text := myConfId.proxy;
  edtProxyPort.Text := myConfId.port;

  // 3. User
  edtUsername.Text := myUserId.username;
  edtPassword.Text := myUserId.password;
  edtRealname.Text := myUserid.realname;
  edtHomepage.Text := myUserid.homepage_url;
  edtEmail.Text    := myUserid.email;

  // 4. Configuration
  cbRunOnlyIdle.Checked := myConfId.run_only_when_idle;
  edtMaxComputations.Value := MyConfId.max_computations;
  edtMaxServices.Value := myConfId.max_services;
  edtMaxDownloads.Value := myConfId.max_downloads;
  edtMaxUploads.Value := myConfId.max_uploads;
end;

function  TParametersForm.validateConfiguration : Boolean;
begin
  Result := true;
end;

procedure TParametersForm.writeConfiguration;
begin
  // 1. Node
  myGPUId.Nodename := edtNodename.Text;
  myGPUId.Region := edtRegion.Text;
  myGPUID.Street := edtStreet.Text;
  myGPUID.Zip := edtZip.Text;
  myGPUId.City := edtCity.Text;
  myGPUID.Description := edtDescription.Text;

  myGPUId.Team := cbTeam.Text;

  //TODO: make it work for all adjacent components
   myGPUId.Longitude := StrToFloat(edtLonDegrees.Text);
   myGPUId.Latitude  := StrToFloat(edtLatDegrees.Text);

  // 2. Connection
  myConfId.proxy := edtProxyHost.Text;
  myConfId.port := edtProxyPort.Text;

  // 3. User
  myUserId.username := edtUsername.Text;
  myUserId.password := edtPassword.Text;
  myUserid.realname := edtRealname.Text;
  myUserid.homepage_url := edtHomepage.Text;
  myUserid.email := edtEmail.Text;

  // 4. Configuration
  myConfId.run_only_when_idle := cbRunOnlyIdle.Checked;
  MyConfId.max_computations := edtMaxComputations.Value;
  myConfId.max_services := edtMaxServices.Value;
  myConfId.max_downloads := edtMaxDownloads.Value;
  myConfId.max_uploads := edtMaxUploads.Value;
end;


initialization
  {$I parametersforms.lrs}

end.


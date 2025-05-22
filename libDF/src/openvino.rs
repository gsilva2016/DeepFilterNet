use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
#[cfg(feature = "timings")]
use std::time::Instant;

use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use ini::Ini;
use ndarray::{prelude::*, Axis};
use tar::Archive;
use tract_core::internal::tract_itertools::izip;
use tract_core::internal::tract_smallvec::alloc::collections::VecDeque;
use tract_core::ops;
use tract_core::prelude::*;
use tract_onnx::{prelude::*, tract_hir::shapefactoid};
use tract_pulse::{internal::ToDim, model::*};
use std::io::prelude::*;
use std::ffi::OsStr;
use crate::*;
//use std::mem;
use lazy_static::lazy_static; 
use std::sync::Mutex;
use ::openvino::
{
    Node, CompiledModel, prepostprocess, Core, DeviceType, ElementType, Layout, ResizeAlgorithm, /*Shape, Tensor,i*/ Model,
};

#[derive(Clone)]
pub struct DfParams {
    config: Ini,
    enc_xml: Vec<u8>,
    enc_bin: Vec<u8>,
    erb_dec_xml: Vec<u8>,
    erb_dec_bin: Vec<u8>,
    df_dec_xml: Vec<u8>,
    df_dec_bin: Vec<u8>,
}

impl DfParams {
    pub fn new(compressed_file: PathBuf) -> Result<Self> {
        panic!("DId I make ithere??? in DfP New");
        let file_name = compressed_file.clone();
        let file_name_ostr = file_name.as_os_str();
        let file_name_str = file_name_ostr.to_str().unwrap();
        let file = File::open(compressed_file).context("Could not open model compressed file.")?;
        if file_name_str.ends_with("zip") {
            Self::from_zip(file)
        } else {
            panic!("OpenVINO models couldn't be found!")
        }
    }
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let zip_bytes = Cursor::new(buf);
        Self::from_zip(zip_bytes)
    }
/*
    fn from_targz<R: Read>(f: R) -> Result<Self> {
        let tar = GzDecoder::new(f);
        let mut archive = Archive::new(tar);
        let mut enc = Vec::new();
        let mut erb_dec = Vec::new();
        let mut df_dec = Vec::new();
        let mut config = Ini::new();
        let mut enc_xml = Vec::new();
        let mut enc_bin = Vec::new();
        for e in archive.entries().context("Could not extract models from tar file.")? {
            let mut file = e.context("Could not open model tar entry.")?;
            let path = file.path().unwrap();
            if path.ends_with("enc.onnx") {
                file.read_to_end(&mut enc)?;
            } else if path.ends_with("erb_dec.onnx") {
                file.read_to_end(&mut erb_dec)?;
            } else if path.ends_with("df_dec.onnx") {
                file.read_to_end(&mut df_dec)?;
            } else if path.ends_with("config.ini") {
                config =
                    Ini::read_from(&mut file).context("Could not load config from tar file.")?;
            } else if path.ends_with("version.txt") {
                let mut version = String::new();
                file.read_to_string(&mut version).expect("Could not read version.txt");
                log::info!("Loading model with id: {}", version);
            } else {
                log::warn!("Found non-matching item in model tar file: {:?}", path)
            }
        }
        Ok(Self {
            config,
            enc,
            erb_dec,
            df_dec,
            enc_xml,
            enc_bin
        })
    }
*/
    fn from_zip<R: Read+Seek>(f: R) -> Result<Self> {
        //let reader = Read::new(f);
//        use zip::HasZipMetadata;
        let mut archive = zip::ZipArchive::new(f)?;
        let mut enc_xml = Vec::new();
        let mut enc_bin = Vec::new();
        let mut erb_dec_xml = Vec::new();
        let mut erb_dec_bin = Vec::new();
        let mut df_dec_xml = Vec::new();
        let mut df_dec_bin = Vec::new();
        let mut config = Ini::new();

        let mut configFile = File::open("/home/intel-admin/gsi/noise-suppression/DeepFilterNet/models/config.ini")?;
        config = Ini::read_from(&mut configFile).context("Could not load config file.")?;
        for e in 0..archive.len() { 
            let mut file = archive.by_index(e)?;
            let path = file.enclosed_name().unwrap();
            log::debug!("Found zip item: {:?}", path);

            if path.ends_with("enc.xml") {
                file.read_to_end(&mut enc_xml)?;
            } else if path.ends_with("enc.bin") {
                file.read_to_end(&mut enc_bin)?;
            } else if path.ends_with("erb_dec.xml") {
                file.read_to_end(&mut erb_dec_xml)?;
            } else if path.ends_with("erb_dec.bin") {
                file.read_to_end(&mut erb_dec_bin)?;
            } else if path.ends_with("df_dec.xml") {
                file.read_to_end(&mut df_dec_xml)?;
            } else if path.ends_with("df_dec.bin") {
                file.read_to_end(&mut df_dec_bin)?;
            } else if path.ends_with("config.ini") {
                config = Ini::read_from(&mut file).context("Could not load config from tar file.")?;
            } else if path.ends_with("version.txt") {
                let mut version = String::new();
                file.read_to_string(&mut version).expect("Could not read version.txt");
                log::info!("Loading model with id: {}", version);
            } else {
                log::warn!("Found non-matching item in model tar file: {:?}", path)
            }
        }
        Ok(Self {
            config,
            enc_xml,
            enc_bin,
            erb_dec_xml,
            erb_dec_bin,
            df_dec_xml,
            df_dec_bin
        })
    }
}
impl Default for DfParams {
    #[allow(unreachable_code)]
    fn default() -> Self {
        #[cfg(feature = "default-model-ov")]
        {
            log::info!("Loading OpenVINO optimized DeepFilternet3.zip");
            return DfParams::from_bytes(include_bytes!(
                "../../models/deepfilternet3-ov.zip"
            ))
            .expect("Could not load OpenVINO model config");
        }
        #[cfg(not(feature = "default-model-ov"))]
        panic!("Not compiled with a default model")
    }
}

#[derive(Clone)]
pub enum ReduceMask {
    NONE = 0,
    MAX = 1,
    MEAN = 2,
}
impl Default for ReduceMask {
    fn default() -> Self {
        ReduceMask::NONE
    }
}
impl TryFrom<i32> for ReduceMask {
    type Error = ();

    fn try_from(v: i32) -> Result<Self, Self::Error> {
        match v {
            x if x == ReduceMask::NONE as i32 => Ok(ReduceMask::NONE),
            x if x == ReduceMask::MAX as i32 => Ok(ReduceMask::MAX),
            x if x == ReduceMask::MEAN as i32 => Ok(ReduceMask::MEAN),
            _ => Err(()),
        }
    }
}
pub struct RuntimeParams {
    pub n_ch: usize,
    pub post_filter: bool,
    pub post_filter_beta: f32,
    pub atten_lim_db: f32,
    pub min_db_thresh: f32,
    pub max_db_erb_thresh: f32,
    pub max_db_df_thresh: f32,
    pub reduce_mask: ReduceMask,
}
impl RuntimeParams {
    pub fn new(
        n_ch: usize,
        post_filter_beta: f32,
        atten_lim_db: f32,
        min_db_thresh: f32,
        max_db_erb_thresh: f32,
        max_db_df_thresh: f32,
        reduce_mask: ReduceMask,
    ) -> Self {
        let post_filter = post_filter_beta > 0.;
        Self {
            n_ch,
            post_filter,
            post_filter_beta,
            atten_lim_db,
            min_db_thresh,
            max_db_erb_thresh,
            max_db_df_thresh,
            reduce_mask,
        }
    }
    pub fn with_post_filter(mut self, beta: f32) -> Self {
        assert!(beta >= 0.); // Cannot be negative
        if beta > 0. {
            self.post_filter = true;
        }
        self.post_filter_beta = beta;
        self
    }
    pub fn with_atten_lim(mut self, atten_lim_db: f32) -> Self {
        self.atten_lim_db = atten_lim_db;
        self
    }
    pub fn with_thresholds(
        mut self,
        min_db_thresh: f32,
        max_db_erb_thresh: f32,
        max_db_df_thresh: f32,
    ) -> Self {
        self.min_db_thresh = min_db_thresh;
        self.max_db_erb_thresh = max_db_erb_thresh;
        self.max_db_df_thresh = max_db_df_thresh;
        self
    }
    pub fn with_mask_reduce(mut self, red: ReduceMask) -> Self {
        self.reduce_mask = red;
        self
    }
    pub fn default_with_ch(channels: usize) -> Self {
        RuntimeParams {
            n_ch: channels,
            post_filter: false,
            post_filter_beta: 0.02,
            atten_lim_db: 100.,
            min_db_thresh: -10.,
            max_db_erb_thresh: 30.,
            max_db_df_thresh: 20.,
            reduce_mask: ReduceMask::NONE, // TODO: ReduceMask::MEAN,
        }
    }
}
impl Default for RuntimeParams {
    fn default() -> Self {
        Self::default_with_ch(1)
    }
}

/*
struct DfOpenVINO {
    core: Core,
    enc: CompiledModel,
    erb_dec: CompiledModel,
    df_dec: CompiledModel
}
*/
/*
impl Clone for DfOpenVINO {
    fn clone(&self) -> Self {
        DfOpenVINO {
            core: Box::new(*self.core), // copy ptrs
            enc: self.enc,
            erb_dec: self.erb_dec,
            df_dec: self.df_dec
        }
    }
}
*/
/*
impl DfOpenVINO {
    pub fn new(core: &Core, enc: &Model, erb_dec: &Model, df_dec: &Model) -> Self {
        //let core = Core::new();

        Self 
        {
            core: *core,//.expect("Could not load new Core()"),
            enc: *enc, //Box::new(enc),
            erb_dec: *erb_dec, //Box::new(erb_dec),
           df_dec: *df_dec //Box::new(df_dec)
        }
    }

}
*/

lazy_static! {
    static ref CORE: Mutex<Core> = Mutex::new(Core::new().expect("Unable to load Core"));
}

//static mut ENC: Option<&CompiledModel> = None;
static mut ENC: Option<Arc<Mutex<CompiledModel>>> = None;
static mut ERB_DEC: Option<Arc<Mutex<CompiledModel>>> = None;
static mut DF_DEC: Option<Arc<Mutex<CompiledModel>>> = None;
//lazy_static! {
//    static ref ENC: Mutex<CompiledModel> = Mutex::new(CompileModel::from_ptr());
//}

//
#[derive(Clone)]
pub struct DfTract {
//    core: Core,
//    enc: CompiledModel,
//    enc_e0: Node,
//    enc_e1: Node,
//    enc_e2: Node,
//    enc_e3: Node,
//    enc_emb: Node,
//    enc_c0: Node,
//    enc_lsnr: Node,
//    erb_dec: CompiledModel,
//    erb_dec_m: Node,
//    df_dec: CompiledModel,    
//    df_dec_coefs: Node,
//    df_dec_235: Node,
    pub lookahead: usize,
    pub df_lookahead: usize,
    pub conv_lookahead: usize,
    pub sr: usize,
    pub ch: usize,
    pub fft_size: usize,
    pub hop_size: usize,
    pub nb_erb: usize,
    pub min_nb_erb_freqs: usize,
    pub nb_df: usize,
    pub n_freqs: usize,
    pub df_order: usize,
    pub post_filter: bool,
    pub post_filter_beta: f32,
    pub alpha: f32,
    pub min_db_thresh: f32,
    pub max_db_erb_thresh: f32,
    pub max_db_df_thresh: f32,
    pub reduce_mask: ReduceMask,
    pub atten_lim: Option<f32>,
    pub df_states: Vec<DFState>,
    pub spec_buf: Tensor, // Real-valued spectrogram buffer of shape [n_ch, 1, 1, n_freqs, 2]
    erb_buf: TValue,      // Real-valued ERB feature buffer of shape [n_ch, 1, 1, n_erb]
    cplx_buf: TValue,     // Real-valued complex epectrum shape for DF of shape [n_ch, 1, nb_df, 2]
    m_zeros: Vec<f32>,    // Preallocated buffer for applying a zero mask
    rolling_spec_buf_y: VecDeque<Tensor>, // Enhanced stage 1 spec buf
    rolling_spec_buf_x: VecDeque<Tensor>, // Noisy spec buf
    skip_counter: usize,  // Increment when wanting to skip processing due to low RMS
}

#[cfg(all(not(feature = "capi"), feature = "default-model"))]
impl Default for DfTract {
    fn default() -> Self {
        let r_params = RuntimeParams::default();
        let df_params = DfParams::default();
        DfTract::new(df_params, &r_params).expect("Could not load DfTract")
    }
}

impl DfTract {
    pub fn new(dfp: DfParams, rp: &RuntimeParams) -> Result<Self> {
        #[cfg(feature = "timings")]
        let t0 = Instant::now();        
        let config = dfp.config;
        let model_cfg = config.section(Some("deepfilternet")).unwrap();
        let df_cfg = config.section(Some("df")).unwrap();
        let ch = rp.n_ch;
        let hop_size = df_cfg.get("hop_size").unwrap().parse::<usize>()?;

        // load enc model
        log::warn!("Start ENC MODE Compile");
        //let mut core = CORE.lock().unwrap();
        log::warn!("Core created....");
        let mut ovm = CORE.lock().unwrap() 
            .read_model_from_file(
                "/home/intel-admin/gsi/noise-suppression/DeepFilterNet/models/enc-static.xml",
                "/home/intel-admin/gsi/noise-suppression/DeepFilterNet/models/enc-static.bin"
            )?;
        log::warn!("ovm created....");
        let tport = ovm.get_output_by_index(0)?;
        log::warn!("xmodel output found: {} / num inputs: {}", tport.get_name()?, ovm.get_inputs_len()?);

//        let mut core2 = Core::new().unwrap();
 //       log::warn!("created new core...");
        let mut enc = CORE.lock().unwrap()
            .compile_model(&ovm, DeviceType::CPU).unwrap();
        log::warn!("ENC model compiled!");
        /*
        let enc = init_encoder_from_read(
            &dfp.enc_xml,
            &dfp.enc_bin,
            df_cfg, 
            ch, 
            &mut CORE.lock().unwrap()); //.expect("Unable to load enc model.");
        */
        //get_output_by_name
        let enc_e0 = enc.get_output_by_index(0)?;
        let enc_e1 = enc.get_output_by_index(1)?;
        let enc_e2 = enc.get_output_by_index(2)?;
        let enc_e3 = enc.get_output_by_index(3)?;
        let enc_emb = enc.get_output_by_index(4)?;
        let enc_c0 = enc.get_output_by_index(5)?;
        let enc_lsnr = enc.get_output_by_index(6)?;

        unsafe {
            ENC = Some(Arc::new(Mutex::new(enc)));
        }


//        let dfOV = DfOpenVINO::new(&enc);
//        dfOV.enc = Some(Box::new(enc)?);

        let erb_dec = init_erb_decoder_from_read(
            &dfp.erb_dec_xml,
            &dfp.erb_dec_bin,
            model_cfg,
            df_cfg,
            ch,
            Some(rp.reduce_mask.clone()),
            &mut CORE.lock().unwrap()).expect("Unable to load erb_dec model.");
//        let erb_dec_m = erb_dec.get_output_by_index(0)?;
        unsafe {
            ERB_DEC = Some(Arc::new(Mutex::new(erb_dec)));
        }

        let df_dec = init_df_decoder_from_read(
            &dfp.df_dec_xml,
            &dfp.df_dec_bin,
            model_cfg, 
            df_cfg, 
            ch,
            &mut CORE.lock().unwrap()).expect("Unable to load df_enc model.");
//        let df_dec_coefs = df_dec.get_output_by_index(0)?;
//        let df_dec_235 = df_dec.get_output_by_index(1)?;
        unsafe {
            DF_DEC = Some(Arc::new(Mutex::new(df_dec)));
        }

        #[cfg(feature = "timings")]
        let t1 = Instant::now();

        let sr = df_cfg.get("sr").unwrap().parse::<usize>()?;
        let fft_size = df_cfg.get("fft_size").unwrap().parse::<usize>()?;
        let min_nb_erb_freqs = df_cfg.get("min_nb_erb_freqs").unwrap().parse::<usize>()?;
        let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
        let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
        let df_order = df_cfg
            .get("df_order")
            .unwrap_or_else(|| model_cfg.get("df_order").unwrap())
            .parse::<usize>()?;
        let conv_lookahead = model_cfg.get("conv_lookahead").unwrap().parse::<usize>()?;
        let df_lookahead = df_cfg
            .get("df_lookahead")
            .unwrap_or_else(|| model_cfg.get("df_lookahead").unwrap())
            .parse::<usize>()?;
        let n_freqs = fft_size / 2 + 1;
        let alpha = if let Some(a) = df_cfg.get("norm_alpha") {
            a.parse::<f32>()?
        } else {
            let tau = df_cfg.get("norm_tau").unwrap().parse::<f32>()?;
            calc_norm_alpha(sr, hop_size, tau)
        };
        let atten_lim = rp.atten_lim_db.abs();
        let atten_lim = if atten_lim >= 100. {
            None
        } else if atten_lim < 0.01 {
            log::warn!("Attenuation limit too strong. No noise reduction will be performed");
            Some(1.)
        } else {
            log::info!("Running with an attenuation limit of {:.0} dB", atten_lim);
            Some(10f32.powf(-atten_lim / 20.))
        };
        let spec_shape = [1, 1, 1, n_freqs, 2];
        let spec_buf = unsafe { Tensor::uninitialized_dt(f32::datum_type(), &spec_shape)? };
        let erb_buf = TValue::from(unsafe {
            Tensor::uninitialized_dt(f32::datum_type(), &[1, 1, 1, nb_erb])?
        });
        let cplx_buf = TValue::from(unsafe {
            Tensor::uninitialized_dt(f32::datum_type(), &[1, 1, nb_df, 2])?
        });
        let m_zeros = vec![0.; nb_erb];

        let model_type = config.section(Some("train")).unwrap().get("model").unwrap();
        let lookahead = match model_type {
            "deepfilternet2" => bail!(
                "DeepFilterNet2 models are deprecated. Please use version v0.3.1 for these models."
            ),
            "deepfilternet3" => conv_lookahead.max(df_lookahead),
            _ => bail!("Unsupported model type {}", model_type),
        };
        log::info!(
            "Running with model type {} lookahead {}",
            model_type,
            lookahead
        );

        let rolling_spec_buf_y = VecDeque::with_capacity(df_order + lookahead);
        let rolling_spec_buf_x = VecDeque::with_capacity(lookahead.max(df_order));

        let mut state = DFState::new(sr, fft_size, hop_size, nb_erb, min_nb_erb_freqs);
        state.init_norm_states(nb_df);
        let df_states = vec![state];

        let mut m = Self {
 //           dfOV,
            //core,
//            enc,
//            enc_e0,
//            enc_e1,
//            enc_e2,
//            enc_e3,
//            enc_emb,
//            enc_c0,
//            enc_lsnr,
//            erb_dec,
//            erb_dec_m,
//            df_dec,
//            df_dec_coefs,
//            df_dec_235,
            lookahead,
            conv_lookahead,
            df_lookahead,
            sr,
            ch,
            fft_size,
            hop_size,
            nb_erb,
            min_nb_erb_freqs,
            nb_df,
            n_freqs,
            df_order,
            alpha,
            min_db_thresh: rp.min_db_thresh,
            max_db_erb_thresh: rp.max_db_erb_thresh,
            max_db_df_thresh: rp.max_db_df_thresh,
            reduce_mask: rp.reduce_mask.clone(),
            atten_lim,
            spec_buf,
            erb_buf,
            cplx_buf,
            m_zeros,
            rolling_spec_buf_y,
            rolling_spec_buf_x,
            df_states,
            post_filter: rp.post_filter,
            post_filter_beta: rp.post_filter_beta,
            skip_counter: 0,
        };
        m.init()?;
        #[cfg(feature = "timings")]
        log::info!(
            "Init DfTract in {:.2}ms (models in {:.2}ms)",
            t0.elapsed().as_secs_f32() * 1000.,
            (t1 - t0).as_secs_f32() * 1000.
        );

        Ok(m)
    }

    pub fn set_pf_beta(&mut self, beta: f32) {
        log::debug!("Setting post-filter beta to {beta}");
        self.post_filter_beta = beta;
        if beta > 0. {
            self.post_filter = true;
        } else if beta == 0. {
            self.post_filter = false;
        } else {
            log::warn!("Post-filter beta cannot be smaller than 0.");
            self.post_filter = false;
            self.post_filter_beta = 0.;
        }
    }

    pub fn set_atten_lim(&mut self, db: f32) {
        let lim = db.abs();
        self.atten_lim = if lim >= 100. {
            None
        } else if lim < 0.01 {
            log::warn!("Attenuation limit too strong. No noise reduction will be performed");
            Some(1.)
        } else {
            log::debug!("Setting attenuation limit to {:.1} dB", lim);
            Some(10f32.powf(-lim / 20.))
        };
    }

    pub fn init(&mut self) -> Result<()> {
        let ch = self.ch;
        let spec_shape = [ch, 1, 1, self.n_freqs, 2];
        self.rolling_spec_buf_y.clear();
        for _ in 0..(self.df_order + self.conv_lookahead) {
            self.rolling_spec_buf_y
                .push_back(tensor0(0f32).broadcast_scalar_to_shape(&spec_shape)?);
        }
        for _ in 0..self.df_order.max(self.lookahead) {
            self.rolling_spec_buf_x
                .push_back(tensor0(0f32).broadcast_scalar_to_shape(&spec_shape)?);
        }
        if ch > self.df_states.len() {
            for _ in self.df_states.len()..ch {
                let mut state = DFState::new(
                    self.sr,
                    self.fft_size,
                    self.hop_size,
                    self.nb_erb,
                    self.min_nb_erb_freqs,
                );
                state.init_norm_states(self.nb_df);
                self.df_states.push(state)
            }
        }
        self.spec_buf = Tensor::zero::<f32>(&spec_shape)?;
        self.erb_buf = TValue::from(Tensor::zero::<f32>(&[ch, 1, 1, self.nb_erb])?);
        self.cplx_buf = TValue::from(Tensor::zero::<f32>(&[ch, 1, self.nb_df, 2])?);

        Ok(())
    }

    /// Process a FD sample and return the raw gains and DF coefs.
    ///
    /// Warning:
    ///     `self.spec_buf` needs to be initialized correctly before calling this method!
    ///
    /// Returns:
    ///     - lsnr: Local SNR estiamte.
    ///     - gains: Gain estimates of shape `[n_ch, 1, 1, n_erb]`.
    ///     - coefs: Real-valued DF coefficients estimates of shape `[n_ch, 1, 1, n_erb, 2]`.
    pub fn process_raw(&mut self) -> Result<(f32, Option<Vec<f32>>, Option<Vec<f32>>)> {
        let spec = self.spec_buf.to_array_view()?;
        let ch = spec.len_of(Axis(0));

        for (nsy_ch, mut erb_ch, mut cplx_ch, state) in izip!(
            spec.axis_iter(Axis(0)),
            tvalue_to_array_view_mut(&mut self.erb_buf).axis_iter_mut(Axis(0)),
            tvalue_to_array_view_mut(&mut self.cplx_buf).axis_iter_mut(Axis(0)),
            self.df_states.iter_mut()
        ) {
            let nsy_ch = as_slice_complex(nsy_ch.as_slice().unwrap());
            state.feat_erb(nsy_ch, self.alpha, erb_ch.as_slice_mut().unwrap());
            state.feat_cplx(
                &nsy_ch[..self.nb_df],
                self.alpha,
                as_slice_mut_complex(cplx_ch.as_slice_mut().unwrap()),
            );
        }
        // Run encoder
        let mut infer_request = None;
        unsafe {
            infer_request = Some(ENC.as_ref().unwrap().lock().unwrap().create_infer_request()?);
        }
        let input_shape = ::openvino::Shape::new(&[1,1,self.hop_size as i64,32])?;
        let element_type = ElementType::F32;
        let mut tensor_featerb = ::openvino::Tensor::new(element_type, &input_shape)?;
        let buffer = tensor_featerb.get_raw_data_mut()?;
        unsafe {
            buffer.copy_from_slice(self.erb_buf.as_bytes());
        }
        let mut infer_request = infer_request.expect("");
        infer_request.set_tensor("feat_erb", &tensor_featerb)?;

        let input_shape_spec = ::openvino::Shape::new(&[1,2,self.hop_size as i64,96])?;
        let mut tensor_featspec = ::openvino::Tensor::new(element_type, &input_shape_spec)?;
        let buffer_spec = tensor_featspec.get_raw_data_mut()?;
        unsafe {
            buffer_spec.copy_from_slice(self.cplx_buf.as_bytes());
        }
        infer_request.set_tensor("feat_spec", &tensor_featspec)?;
        infer_request.infer()?;
        let emb = infer_request.get_tensor("emb".to_string().as_str())?; //&self.enc_emb.get_name()?)?;
//            .get_data::<f32>()?.to_vec();
        let lsnr = infer_request.get_tensor("lsnr".to_string().as_str())? //&self.enc_lsnr.get_name()?)?
            .get_data::<f32>()?.to_vec(); 
        debug_assert_eq!(lsnr.len(), 1);
        let lsnr = lsnr[0];
        let c0 = infer_request.get_tensor("c0".to_string().as_str())?; //&self.enc_c0.get_name()?)?;
//            .get_data::<f32>()?.to_vec(); 
        let e0 = infer_request.get_tensor("e0".to_string().as_str())?;//&self.enc_e0.get_name()?)?;
        let e1 = infer_request.get_tensor("e1".to_string().as_str())?; //&self.enc_e1.get_name()?)?;
//            .get_data::<f32>()?.to_vec();
        let e2 = infer_request.get_tensor("e2".to_string().as_str())?; //&self.enc_e2.get_name()?)?;
//            .get_data::<f32>()?.to_vec();
        let e3 = infer_request.get_tensor("e3")?; //&self.enc_e3.get_name()?)?;
//            .get_data::<f32>()?.to_vec();

        let (apply_gains, apply_gain_zeros, apply_df) = self.apply_stages(lsnr);

        log::trace!(
            "Enhancing frame with lsnr {:>5.1} dB. Applying stage 1: {} and stage 2: {}.",
            lsnr,
            apply_gains,
            apply_df
        );

        let m = if apply_gains {
            let mut infer_request = None;
            unsafe {
                infer_request = Some(ERB_DEC.as_ref().unwrap().lock().unwrap().create_infer_request()?);
            }
            let mut infer_request = infer_request.expect("Missing ERB_DEC");
//            let mut infer_request = self.erb_dec.create_infer_request()?;
            infer_request.set_tensor("emb", &emb)?; // Clone needed?
            infer_request.set_tensor("e3", &e3)?;
            infer_request.set_tensor("e2", &e2)?;
            infer_request.set_tensor("e1", &e1)?;
            infer_request.set_tensor("e0", &e0)?;
            infer_request.infer()?;
            let mut m = infer_request.get_tensor("c0".to_string().as_str())? //&self.enc_c0.get_name()?)?
                .get_data::<f32>()?.to_vec();

            m.remove(1);
            m.remove(1);
            Some(m)
        } else if apply_gain_zeros {
            // TODO: move outside of loop
//            let data = ?;
            let input_shape = ::openvino::Shape::new(&[self.ch as i64, self.nb_erb as i64])?;
            let element_type = ElementType::F32;
            let mut tensor = ::openvino::Tensor::new(element_type, &input_shape)?;
            let buffer = tensor.get_raw_data_mut()?;
            buffer.fill(0); //.copy_from_slice(&data);
            Some(tensor.get_data::<f32>()?.to_vec())
        } else {
            None
        };

        let coefs = if apply_df {
            let mut infer_request_df = None; 
            unsafe {
                infer_request_df = Some(DF_DEC.as_ref().unwrap().lock().unwrap().create_infer_request()?);
            }
            let mut infer_request = infer_request_df.expect("Missing DF_DEC");
//            let mut infer_request = self.df_dec.create_infer_request()?;
            infer_request.set_tensor("emb", &emb)?;
            infer_request.set_tensor("c0", &c0)?;
            infer_request.infer()?;
            let coefs_shape = ::openvino::Shape::new(&[ch as i64, self.nb_df as i64, self.df_order as i64,2])?;
            let mut coefs = infer_request.get_tensor("coefs")? //&self.df_dec_coefs.get_name()?)?
                .set_shape(&coefs_shape)?
                .get_data::<f32>()?.to_vec();
            Some(coefs)
        } else {
            None
        };

        Ok((lsnr, m, coefs))
    }

    fn demo<T, const N: usize>(v: Vec<T>) -> [T; N] {
        v.try_into()
            .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len())            )
    }

    /// Process a noisy time domain sample and return the enhanced sample via mutable argument.
    pub fn process(&mut self, noisy: ArrayView2<f32>, mut enh: ArrayViewMut2<f32>) -> Result<f32> {
        debug_assert_eq!(noisy.len_of(Axis(0)), enh.len_of(Axis(0)));
        debug_assert_eq!(noisy.len_of(Axis(1)), enh.len_of(Axis(1)));
        debug_assert_eq!(noisy.len_of(Axis(1)), self.hop_size);
        let (max_a, e) = noisy.iter().fold((0f32, 0f32), |acc, x| {
            (acc.0.max(x.abs()), acc.1 + x.powi(2))
        });
        let rms = e / noisy.len() as f32;
        if rms < 1e-7 {
            self.skip_counter += 1;
        } else {
            self.skip_counter = 0;
        }
        if self.skip_counter > 5 {
            enh.fill(0.);
            return Ok(-15.);
        }
        if max_a > 0.9999 {
            log::warn!("Possible clipping detected ({:.3}).", max_a)
        }

        // Signal model: y = f(s + n) = f(x)
        self.rolling_spec_buf_y.pop_front();
        self.rolling_spec_buf_x.pop_front();
        for (ns_ch, mut rbuf, state) in izip!(
            noisy.axis_iter(Axis(0)),
            self.spec_buf.to_array_view_mut()?.axis_iter_mut(Axis(0)),
            self.df_states.iter_mut(),
        ) {
            let spec = as_slice_mut_complex(rbuf.as_slice_mut().unwrap());
            state.analysis(ns_ch.as_slice().unwrap(), spec);
        }
        self.rolling_spec_buf_y.push_back(self.spec_buf.clone());
        self.rolling_spec_buf_x.push_back(self.spec_buf.clone());
        if self.atten_lim.unwrap_or_default() == 1. {
            enh.assign(&noisy);
            return Ok(35.);
        }

        let (lsnr, gains, coefs) = self.process_raw()?;

        let (apply_erb, _, _) = self.apply_stages(lsnr);
        let mut spec = self
            .rolling_spec_buf_y
            .get_mut(self.df_order - 1)
            .unwrap()
            .to_array_view_mut()?;
        if let Some(gains) = gains {
            let mut gains = ndarray::Array::from_vec(gains);
//            let mut gains = demo(gains) //gains.into_array()?;
            if gains.shape()[0] < noisy.shape()[0] {
                // Mask was reduced to single channel
                let gain_slc = gains.as_slice_mut().unwrap();
                for mut spec_ch in spec.axis_iter_mut(Axis(0)) {
                    self.df_states[0].apply_mask(
                        as_slice_mut_complex(spec_ch.as_slice_mut().unwrap()),
                        gain_slc,
                    );
                }
            } else {
                // Same number of channels of gains and spec
                for (gains_ch, mut spec_ch) in
                    gains.axis_iter(Axis(0)).zip(spec.axis_iter_mut(Axis(0)))
                {
                    let gain_slc = gains_ch.as_slice().unwrap();
                    self.df_states[0].apply_mask(
                        as_slice_mut_complex(spec_ch.as_slice_mut().unwrap()),
                        gain_slc,
                    );
                }
            }
            self.skip_counter = 0;
        } else {
            // gains are None => skipped due to LSNR
            self.skip_counter += 1;
        }

        // This spectrum will only be used for the upper frequecies
        let spec = self.rolling_spec_buf_y.get_mut(self.df_order - 1).unwrap();
        self.spec_buf.clone_from(spec);
        if let Some(coefs) = coefs {
            let mut coefs = ndarray::Array::from_vec(coefs);
            let mut coefs = Tensor::from(coefs);
            df(
                &self.rolling_spec_buf_x,
                coefs,
                self.nb_df,
                self.df_order,
                self.n_freqs,
                &mut self.spec_buf,
            )?;
        };

        let spec_noisy = as_arrayview_complex(
            self.rolling_spec_buf_x
                .get(self.lookahead.max(self.df_order) - self.lookahead - 1)
                .unwrap()
                .to_array_view::<f32>()
                .unwrap(),
            &[self.ch, self.n_freqs],
        )
        .into_dimensionality::<Ix2>()
        .unwrap();
        let mut spec_enh = as_arrayview_mut_complex(
            self.spec_buf.to_array_view_mut::<f32>().unwrap(),
            &[self.ch, self.n_freqs],
        )
        .into_dimensionality::<Ix2>()
        .unwrap();

        // Run post filter
        if apply_erb && self.post_filter {
            post_filter(
                spec_noisy.as_slice().unwrap(),
                spec_enh.as_slice_mut().unwrap(),
                self.post_filter_beta,
            );
        }

        // Limit noise attenuation by mixing back some of the noisy signal
        if let Some(lim) = self.atten_lim {
            spec_enh.map_inplace(|x| *x *= 1. - lim);
            spec_enh.scaled_add(lim.into(), &spec_noisy);
        }

        for (state, spec_ch, mut enh_out_ch) in izip!(
            self.df_states.iter_mut(),
            spec_enh.axis_iter(Axis(0)),
            enh.axis_iter_mut(Axis(0)),
        ) {
            state.synthesis(
                spec_ch.to_owned().as_slice_mut().unwrap(),
                enh_out_ch.as_slice_mut().unwrap(),
            );
        }
        Ok(lsnr)
    }

    /// For some frames, processing may be skipped based on the current local snr and the defined
    /// thresholds. This methods indiciated whether stage 1 (gains) and stage 2 (DF) can be
    /// skipped.
    ///
    /// Args:
    ///     - lsnr: Current local SNR estimate
    ///
    /// Returns:
    ///     - apply_gains: Local SNR is above `min_dfb_erb_thresh`, gains are estimated and should be
    ///         applied
    ///     - apply_gain_zeros: Local SNR is less than `min_db_thresh`, no speech is detected.
    ///         Zeros should be applied instead of the gain estimates.
    ///     - apply_df: Local SNR is greater than `max_db_df_thresh` and the estimated DF coefs
    ///         should be applied
    pub fn apply_stages(&self, lsnr: f32) -> (bool, bool, bool) {
        if lsnr < self.min_db_thresh {
            // Only noise detected, just apply a zero mask
            (false, true, false)
        } else if lsnr > self.max_db_erb_thresh {
            // Clean speech signal detected, don't apply any processing
            (false, false, false)
        } else if lsnr > self.max_db_df_thresh {
            // Only little noisy signal detected, just apply 1st stage, skip DF stage
            (true, false, false)
        } else {
            // Regular noisy signal detected, apply 1st and 2nd stage
            (true, false, true)
        }
    }

    pub fn set_spec_buffer(&mut self, spec: ArrayView2<f32>) -> Result<()> {
        debug_assert_eq!(self.spec_buf.shape(), spec.shape());
        let mut buf = self.spec_buf.to_array_view_mut()?.into_shape([self.ch, self.n_freqs])?;
        for (i_ch, mut b_ch) in spec.outer_iter().zip(buf.outer_iter_mut()) {
            for (&i, b) in i_ch.iter().zip(b_ch.iter_mut()) {
                *b = i
            }
        }
        Ok(())
    }

    pub fn get_spec_noisy(&self) -> ArrayView2<Complex32> {
        as_arrayview_complex(
            self.rolling_spec_buf_x
                .get(self.lookahead.max(self.df_order) - self.lookahead - 1)
                .unwrap()
                .to_array_view::<f32>()
                .unwrap(),
            &[self.ch, self.n_freqs],
        )
        .into_dimensionality::<Ix2>()
        .unwrap()
    }
    pub fn get_spec_enh(&self) -> ArrayView2<Complex32> {
        as_arrayview_complex(
            self.spec_buf.to_array_view::<f32>().unwrap(),
            &[self.ch, self.n_freqs],
        )
        .into_dimensionality::<Ix2>()
        .unwrap()
    }
    pub fn get_mut_spec_enh(&mut self) -> ArrayViewMut2<Complex32> {
        as_arrayview_mut_complex(
            self.spec_buf.to_array_view_mut::<f32>().unwrap(),
            &[self.ch, self.n_freqs],
        )
        .into_dimensionality::<Ix2>()
        .unwrap()
    }
}

/// Deep Filtering.
///
/// Args:
///     - spec: Spectrogram buffer for the corresponding time steps. Needs to contain `df_order + conv_lookahead` frames and applies DF to the oldest frames.
///     - coefs: Complex DF coefficients of shape `[C, N, F', 2]`, `N`: `df_order`, `F'`: `nb_df`
///     - nb_df: Number of DF frequency bins
///     - df_order: Deep Filtering order
///     - n_freqs: Number of FFT bins
///     - spec_out: Ouput buffer of shape `[C, F, 2]`, `F`: `n_freqs`
fn df(
    spec: &VecDeque<Tensor>,
    coefs: Tensor,
    nb_df: usize,
    df_order: usize,
    n_freqs: usize,
    spec_out: &mut Tensor,
) -> Result<()> {
    let ch = spec.back().unwrap().shape()[0];
    debug_assert_eq!(n_freqs, spec.back().unwrap().shape()[3]);
    debug_assert_eq!(n_freqs, spec_out.shape()[3]);
    debug_assert_eq!(ch, coefs.shape()[0]);
    debug_assert_eq!(nb_df, coefs.shape()[1]);
    debug_assert_eq!(df_order, coefs.shape()[2]);
    debug_assert_eq!(ch, spec_out.shape()[0]);
    debug_assert!(spec.len() >= df_order);
    let mut o_f: ArrayViewMut2<Complex32> =
        as_arrayview_mut_complex(spec_out.to_array_view_mut::<f32>()?, &[ch, n_freqs])
            .into_dimensionality()?;
    // Zero relevant frequency bins of output
    o_f.slice_mut(s![.., ..nb_df]).fill(Complex32::default());
    let coefs_arr: ArrayView3<Complex32> =
        as_arrayview_complex(coefs.to_array_view::<f32>()?, &[ch, nb_df, df_order])
            .into_dimensionality()?;
    // Transform spec to an complex array and iterate over time frames of spec and coefs
    let spec_iter = spec.iter().map(|s| {
        as_arrayview_complex(s.to_array_view::<f32>().unwrap(), &[ch, n_freqs])
            .into_dimensionality::<Ix2>()
            .unwrap()
    });
    // Iterate over DF frames
    for (s_f, c_f) in spec_iter.zip(coefs_arr.axis_iter(Axis(2))) {
        // Iterate over channels
        for (s_ch, c_ch, mut o_ch) in
            izip!(s_f.outer_iter(), c_f.outer_iter(), o_f.outer_iter_mut())
        {
            // Apply DF for each frequency bin up to `nb_df`
            for (&s, &c, o) in izip!(s_ch, c_ch, o_ch.iter_mut()) {
                *o += s * c
            }
        }
    }
    Ok(())
}
/*
fn init_encoder_impl(
    mut m: InferenceModel,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<TypedModel> {
    log::debug!("Start init encoder.");
    let s = m.symbol_table.sym("S");

    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let feat_erb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, 1, s, nb_erb));
    let feat_spec = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, 2, s, nb_df));

    log::debug!(
        "Encoder input: \n feat_erb  [{:?}]\n feat_spec [{:?}]",
        feat_erb.shape,
        feat_spec.shape,
    );
    m = m
        .with_input_fact(0, feat_erb)?
        .with_input_fact(1, feat_spec)?
        .with_input_names(["feat_erb", "feat_spec"])?
        .with_output_names(["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"])?;

    m.analyse(true)?;
    let mut m = m.into_typed()?;

    m.declutter()?;
    let pulsed = PulsedModel::new(&m, s, &1.to_dim())?;
    log::info!("Init encoder");
    let m = pulsed.into_typed()?.into_optimized()?;
    Ok(m)
}
fn init_encoder(m: &Path, df_cfg: &ini::Properties, n_ch: usize) -> Result<TypedModel> {
    let m = tract_onnx::onnx().with_ignore_output_shapes(true).model_for_path(m)?;
    init_encoder_impl(m, df_cfg, n_ch)
}
*/
fn init_encoder_from_read(
    m_xml: &[u8],
    m_bin: &[u8],
    df_cfg: &ini::Properties,
    n_ch: usize,
    core: &mut Core,
) -> Result<CompiledModel> {
    //let m = tract_onnx::onnx().with_ignore_output_shapes(true).model_for_read(m)?;
//    log::info!("init_encoder_from_read {}", m_bin.len());
    //log::info!("init_encoder_from {}", hop_size);

//    let mut core2 = Core::new().unwrap(); 
    let ovm = core
        .read_model_from_file(
            "/home/intel-admin/gsi/noise-suppression/DeepFilterNet/models/enc-static.xml",
            "/home/intel-admin/gsi/noise-suppression/DeepFilterNet/models/enc-static.bin",
        )
        .unwrap();
    let mut m = core.compile_model(&ovm, DeviceType::CPU)?;
    log::warn!("ENC model compiled!");
    Ok(m)

/*
    let ovm = core
        .read_model_from_buffer(
            m_xml,
            Some(&weights),
        )
        .unwrap();
    let mut m = core.compile_model(&ovm, DeviceType::CPU)?;
    Ok(m)
*/
}

fn init_erb_decoder_impl(
    m_xml: &[u8],
    m_bin: &[u8],
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
    mask_reduction: Option<ReduceMask>,
    core: &mut Core,
) -> Result<CompiledModel> {
    log::debug!("Load OV ERB decoder.");
    let weights = {
        let weights = m_bin;
        let shape = ::openvino::Shape::new(&[1, weights.len() as i64]).unwrap();
        let mut tensor = ::openvino::Tensor::new(ElementType::U8, &shape).unwrap();
        let buffer = tensor.get_raw_data_mut().unwrap();
        tensor
    };      
    let ovm = core
        .read_model_from_buffer(
            m_xml,
            Some(&weights),
        )
        .unwrap();
    let mut m = core.compile_model(&ovm, DeviceType::CPU)?;

    /** Exported model already knows these shapes
    //let s = m.symbol_table.sym("S");
    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let layer_width = net_cfg.get("conv_ch").unwrap().parse::<usize>()?;
    let n_hidden = layer_width * nb_erb / 4;

    let emb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, s, n_hidden));
    let e3f = nb_erb / 4;
    let e3 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, layer_width, s, e3f));
    let e2 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, layer_width, s, e3f));
    let e1f = nb_erb / 2;
    let e1 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, layer_width, s, e1f));
    let e0 = InferenceFact::dt_shape(
        f32::datum_type(),
        shapefactoid!(n_ch, layer_width, s, nb_erb),
    );
    log::debug!(
        "ERB decoder input: \n emb [{:?}]\n e3  [{:?}]\n e2  [{:?}]\n e1  [{:?}]\n e0  [{:?}]",
        emb.shape,
        e3.shape,
        e2.shape,
        e1.shape,
        e0.shape
    );
    */

    /* need to add AVG op into OV IR...use none for now
    let mut output_name = "m".to_string();

    m = m
        .with_input_fact(0, emb)?
        .with_input_fact(1, e3)?
        .with_input_fact(2, e2)?
        .with_input_fact(3, e1)?
        .with_input_fact(4, e0)?
        .with_input_names(["emb", "e3", "e2", "e1", "e0"])?;
    // .with_output_names([output_name])?;

    m.analyse(true)?;

    let mut m = m.into_typed()?;

    m.declutter()?;
    let pulsed = PulsedModel::new(&m, s, &1.to_dim())?;
    let mut m = pulsed.into_typed()?;
    log::info!("Init ERB decoder");

    if let Some(r) = mask_reduction {
        let outlets = m.output_outlets()?;
        let mask_outlet = outlets[0];
        let ch_axis = 0;
        match r {
            ReduceMask::MAX => {
                output_name = "reduce_mask_max".to_string();
                m.wire_node(
                    "reduce_mask_max",
                    ops::nn::Reduce::new(tvec!(ch_axis), ops::nn::Reducer::Max),
                    &[mask_outlet],
                )?;
            }
            ReduceMask::MEAN => {
                let sum = m.wire_node(
                    "reduce_mask_sum".to_string(),
                    ops::nn::Reduce::new(tvec!(ch_axis), ops::nn::Reducer::Sum),
                    &[mask_outlet],
                )?[0];
                let ch_i = m
                    .add_const(
                        "ch".to_string(),
                        Tensor::from_shape(&[1, 1, 1, 1], &[1. / n_ch as f32])?,
                    )
                    .unwrap();
                output_name = "reduce_mask_div_ch".to_string();
                m.wire_node(
                    "reduce_mask_div_ch",
                    tract_core::ops::math::mul(),
                    &[sum, ch_i],
                )?;
            }
            _ => (),
        }
    }
    m = m.with_output_names(&[output_name])?;

    let m = m.into_optimized()?;
    */

    Ok(m)
}

/*
fn init_erb_decoder(
    m: &Path,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
    mask_reduction: Option<ReduceMask>,
) -> Result<TypedModel> {
    let m = tract_onnx::onnx().with_ignore_output_shapes(true).model_for_path(m)?;
    init_erb_decoder_impl(m, net_cfg, df_cfg, n_ch, mask_reduction)
}
*/
fn init_erb_decoder_from_read(
    //m: &mut dyn Read,
    erb_dec_xml: &[u8],
    erb_dec_bin: &[u8],
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
    mask_reduction: Option<ReduceMask>,
    core: &mut Core,
) -> Result<CompiledModel> {
//    let m = tract_onnx::onnx().with_ignore_output_shapes(true).model_for_read(m)?;
    init_erb_decoder_impl(erb_dec_xml, erb_dec_bin,net_cfg, df_cfg, n_ch, mask_reduction, core)
}

fn init_df_decoder_impl(
    m_xml: &[u8],
    m_bin: &[u8],
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
    core: &mut Core
) -> Result<CompiledModel> {
    log::debug!("Load OV DF decoder.");
    let weights = {
        let weights = m_bin;
        let shape = ::openvino::Shape::new(&[1, weights.len() as i64]).unwrap();
        let mut tensor = ::openvino::Tensor::new(ElementType::U8, &shape).unwrap();
        let buffer = tensor.get_raw_data_mut().unwrap();
        tensor
    };
    let ovm = core
        .read_model_from_buffer(
            m_xml,
            Some(&weights),
        )
        .unwrap();
    let mut m = core.compile_model(&ovm, DeviceType::CPU)?;

    /* All below built into OV IR model
    let s = m.symbol_table.sym("S");
    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let layer_width = net_cfg.get("conv_ch").unwrap().parse::<usize>()?;
    let n_hidden = layer_width * nb_erb / 4;

    let emb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, s, n_hidden));
    let c0 = InferenceFact::dt_shape(
        f32::datum_type(),
        shapefactoid!(n_ch, layer_width, s, nb_df),
    );

    log::debug!(
        "ERB decoder input: \n emb [{:?}]\n c0  [{:?}]",
        emb.shape,
        c0.shape,
    );
    m = m
        .with_input_fact(0, emb)?
        .with_input_fact(1, c0)?
        .with_input_names(["emb", "c0"])?
        .with_output_names(["coefs"])?;

    m.analyse(true)?;
    let mut m = m.into_typed()?;

    m.declutter()?;
    let pulsed = PulsedModel::new(&m, s, &1.to_dim())?;
    log::info!("Init DF decoder");
    let m = pulsed.into_typed()?.into_optimized()?;
    */
    Ok(m)
}
/*
fn init_df_decoder(
    m: &Path,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<TypedModel> {
    let m = tract_onnx::onnx().with_ignore_output_shapes(true).model_for_path(m)?;
    init_df_decoder_impl(m, net_cfg, df_cfg, n_ch)
}
*/

fn init_df_decoder_from_read(
//    m: &mut dyn Read,
    df_dec_xml: &[u8],
    df_dec_bin: &[u8],
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
    core: &mut Core,
) -> Result<CompiledModel> {
//    let m = tract_onnx::onnx().with_ignore_output_shapes(true).model_for_read(m)?;

    init_df_decoder_impl(df_dec_xml, df_dec_bin, net_cfg, df_cfg, n_ch, core)
}

fn calc_norm_alpha(sr: usize, hop_size: usize, tau: f32) -> f32 {
    let dt = hop_size as f32 / sr as f32;
    let alpha = f32::exp(-dt / tau);
    let mut a = 1.0;
    let mut precision = 3;
    while a >= 1.0 {
        a = (alpha * 10i32.pow(precision) as f32).round() / 10i32.pow(precision) as f32;
        precision += 1;
    }
    a
}

pub fn as_slice_complex(buffer: &[f32]) -> &[Complex32] {
    unsafe {
        let ptr = buffer.as_ptr() as *const Complex32;
        let len = buffer.len();
        std::slice::from_raw_parts(ptr, len / 2)
    }
}

#[allow(clippy::needless_pass_by_ref_mut)]
pub fn as_slice_mut_complex(buffer: &mut [f32]) -> &mut [Complex32] {
    unsafe {
        let ptr = buffer.as_ptr() as *mut Complex32;
        let len = buffer.len();
        std::slice::from_raw_parts_mut(ptr, len / 2)
    }
}

#[allow(clippy::needless_pass_by_ref_mut)]
pub fn as_slice_mut_real(buffer: &mut [Complex32]) -> &mut [f32] {
    unsafe {
        let ptr = buffer.as_ptr() as *mut f32;
        let len = buffer.len();
        std::slice::from_raw_parts_mut(ptr, len * 2)
    }
}

pub fn slice_as_arrayview<'a>(
    buffer: &[f32],
    shape: &[usize], // having an explicit shape parameter allows to also squeeze axes.
) -> ArrayViewD<'a, f32> {
    debug_assert_eq!(buffer.len(), shape.iter().product::<usize>());
    unsafe {
        let ptr = buffer.as_ptr(); // as *mut Complex32;
        ArrayViewD::from_shape_ptr(shape, ptr)
    }
}
pub fn mut_slice_as_arrayviewmut<'a>(
    buffer: &mut [f32],
    shape: &[usize], // having an explicit shape parameter allows to also squeeze axes.
) -> ArrayViewMutD<'a, f32> {
    debug_assert_eq!(buffer.len(), shape.iter().product::<usize>());
    unsafe {
        let ptr = buffer.as_mut_ptr(); // as *mut Complex32;
        ArrayViewMutD::from_shape_ptr(shape, ptr)
    }
}
pub fn as_arrayview_complex<'a>(
    buffer: ArrayViewD<'a, f32>,
    shape: &[usize], // having an explicit shape parameter allows to also squeeze axes.
) -> ArrayViewD<'a, Complex32> {
    debug_assert_eq!(buffer.shape().last().unwrap(), &2);
    debug_assert_eq!(buffer.len(), 2 * shape.iter().product::<usize>());
    unsafe {
        let ptr = buffer.as_ptr() as *mut Complex32;
        ArrayViewD::from_shape_ptr(shape, ptr)
    }
}
pub fn as_arrayview_mut_complex<'a>(
    buffer: ArrayViewMutD<'a, f32>,
    shape: &[usize], // having an explicit shape parameter allows to also squeeze axes.
) -> ArrayViewMutD<'a, Complex32> {
    debug_assert_eq!(buffer.shape().last().unwrap(), &2);
    debug_assert_eq!(buffer.len(), 2 * shape.iter().product::<usize>());
    unsafe {
        let ptr = buffer.as_ptr() as *mut Complex32;
        ArrayViewMutD::from_shape_ptr(shape, ptr)
    }
}
pub fn tvalue_to_array_view_mut(x: &mut TValue) -> ArrayViewMutD<f32> {
    unsafe {
        match x {
            TValue::Var(x) => {
                ArrayViewMutD::from_shape_ptr(x.shape(), x.as_ptr_unchecked::<f32>() as *mut f32)
            }
            TValue::Const(x) => {
                ArrayViewMutD::from_shape_ptr(x.shape(), x.as_ptr_unchecked::<f32>() as *mut f32)
            }
        }
    }
}

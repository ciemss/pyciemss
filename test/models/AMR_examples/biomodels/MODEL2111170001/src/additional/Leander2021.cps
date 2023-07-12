<?xml version="1.0" encoding="UTF-8"?>
<!-- generated with COPASI 4.34 (Build 251) (http://www.copasi.org) at 2021-11-12T14:54:49Z -->
<?oxygen RNGSchema="http://www.copasi.org/static/schema/CopasiML.rng" type="xml"?>
<COPASI xmlns="http://www.copasi.org/static/schema" versionMajor="4" versionMinor="34" versionDevel="251" copasiSourcesModified="0">
  <ListOfFunctions>
    <Function key="Function_13" name="Mass action (irreversible)" type="MassAction" reversible="false">
      <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <rdf:Description rdf:about="#Function_13">
   <CopasiMT:is rdf:resource="urn:miriam:obo.sbo:SBO:0000163" />
   </rdf:Description>
   </rdf:RDF>
      </MiriamAnnotation>
      <Comment>
        <body xmlns="http://www.w3.org/1999/xhtml">
<b>Mass action rate law for irreversible reactions</b>
<p>
Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does not include any reverse process that creates the reactants from the products. The change of a product quantity is proportional to the quantity of one reactant.
</p>
</body>
      </Comment>
      <Expression>
        k1*PRODUCT&lt;substrate_i>
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_80" name="k1" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_81" name="substrate" order="1" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_40" name="f" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

      </MiriamAnnotation>
      <Comment>
        f(C,S;K,q)
      </Comment>
      <Expression>
        S/(S+(q*(C/2))+K)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_264" name="S" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_263" name="q" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_262" name="C" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_261" name="K" order="3" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_41" name="rg" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

      </MiriamAnnotation>
      <Expression>
        r*T/(T+KT)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_267" name="r" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_266" name="T" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_265" name="KT" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_42" name="Rate Law for R1" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

      </MiriamAnnotation>
      <Expression>
        alpha*A2
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_269" name="alpha" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_268" name="A2" order="1" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_43" name="Rate Law for R2" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        muA*A1-(rg(r,T,KT)*A1)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_270" name="muA" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_250" name="A1" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_271" name="r" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_272" name="T" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_273" name="KT" order="4" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_44" name="Rate Law for R5" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        alpha*f(F,q,A,Kf)*Substrate
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_278" name="alpha" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_277" name="F" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_276" name="q" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_275" name="A" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_274" name="Kf" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_279" name="Substrate" order="5" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_45" name="Rate Law for R3" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (r2*(1-(At/Ka))*A2pos)+(a2*A2neg)+mu*A2posact
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_285" name="r2" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_284" name="At" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_283" name="Ka" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_282" name="A2pos" order="3" role="product"/>
        <ParameterDescription key="FunctionParameter_281" name="a2" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_280" name="A2neg" order="5" role="modifier"/>
        <ParameterDescription key="FunctionParameter_286" name="mu" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_287" name="A2posact" order="7" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_46" name="Rate Law for R4" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (rg(r,T,KT)*A2pos)+(a+a2+muA)*A2pos
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_295" name="r" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_294" name="T" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_293" name="KT" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_292" name="A2pos" order="3" role="substrate"/>
        <ParameterDescription key="FunctionParameter_291" name="a" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_290" name="a2" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_289" name="muA" order="6" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_47" name="Rate Law for R6" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        beta*f(V,(qV/C1),A2pos,Kv)*A2pos
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_301" name="beta" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_300" name="V" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_299" name="qV" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_298" name="C1" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_297" name="A2pos" order="4" role="substrate"/>
        <ParameterDescription key="FunctionParameter_296" name="Kv" order="5" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_48" name="Rate Law for R7" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (r2*(1-(AT/Ka))*A2neg)+(a2pos*A2pos)+(mu*A2negact)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_306" name="r2" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_305" name="AT" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_304" name="Ka" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_303" name="A2neg" order="3" role="product"/>
        <ParameterDescription key="FunctionParameter_302" name="a2pos" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_288" name="A2pos" order="5" role="modifier"/>
        <ParameterDescription key="FunctionParameter_307" name="mu" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_308" name="A2negact" order="7" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_49" name="Rate Law for R8" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        ((a+a2neg+muA)*A2neg) + (rg(r,T,Kt)*A2neg)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_316" name="a" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_315" name="a2neg" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_314" name="muA" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_313" name="A2neg" order="3" role="substrate"/>
        <ParameterDescription key="FunctionParameter_312" name="r" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_311" name="T" order="5" role="modifier"/>
        <ParameterDescription key="FunctionParameter_310" name="Kt" order="6" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_50" name="Rate Law for R9" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        alpha*f(F,q,A,Kf)*A2neg
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_322" name="alpha" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_321" name="F" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_320" name="q" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_319" name="A" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_318" name="Kf" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_317" name="A2neg" order="5" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_51" name="Rate Law for R10" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

      </MiriamAnnotation>
      <Expression>
        ((u+muA+a)*A2posact)+(rg(r,T,KT)*A2posact)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_327" name="u" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_326" name="muA" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_325" name="a" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_324" name="A2posact" order="3" role="substrate"/>
        <ParameterDescription key="FunctionParameter_323" name="r" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_309" name="T" order="5" role="modifier"/>
        <ParameterDescription key="FunctionParameter_328" name="KT" order="6" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_52" name="Rate Law for R12" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

      </MiriamAnnotation>
      <Expression>
        ((mu1+muA)*I)+(rg(r,T,Kt)*I)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_335" name="I" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_334" name="mu1" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_333" name="muA" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_332" name="r" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_331" name="T" order="4" role="modifier"/>
        <ParameterDescription key="FunctionParameter_330" name="Kt" order="5" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_53" name="Rate Law for R13" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

      </MiriamAnnotation>
      <Expression>
        (alpha*f(F,q,A,Kf)*I)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_340" name="alpha" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_339" name="F" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_338" name="q" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_337" name="A" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_336" name="Kf" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_329" name="I" order="5" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_54" name="Rate Law for R14" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

      </MiriamAnnotation>
      <Expression>
        ((mu1+muA)*Iact)+rg(r,T,Kt)*Iact
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_346" name="mu1" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_345" name="muA" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_344" name="Iact" order="2" role="substrate"/>
        <ParameterDescription key="FunctionParameter_343" name="r" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_342" name="T" order="4" role="modifier"/>
        <ParameterDescription key="FunctionParameter_341" name="Kt" order="5" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_55" name="Rate Law for R15" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (mu1+muA)*(I+Iact)+rg(r,T,Kt)*(I+Iact)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_352" name="mu1" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_351" name="muA" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_350" name="I" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_349" name="Iact" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_348" name="r" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_347" name="T" order="5" role="modifier"/>
        <ParameterDescription key="FunctionParameter_353" name="Kt" order="6" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_56" name="Rate Law for R16" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (km0*(M/C1))+(Km*(Mact/C1))
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_360" name="km0" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_359" name="M" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_358" name="C1" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_357" name="Km" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_356" name="Mact" order="4" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_57" name="Rate Law for R17" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (1/C2)*((pf2*I)+(pf2*Iact)+(pf1*Mact))
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_363" name="C2" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_362" name="pf2" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_361" name="I" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_354" name="Iact" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_355" name="pf1" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_364" name="Mact" order="5" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_58" name="Rate Law for R19" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (px/C2)*((I+Iact+A2act+Mact))
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_370" name="px" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_369" name="C2" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_368" name="I" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_367" name="Iact" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_366" name="A2act" order="4" role="modifier"/>
        <ParameterDescription key="FunctionParameter_365" name="Mact" order="5" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_59" name="Rate Law for R21" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        pT*Mact
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_376" name="pT" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_375" name="Mact" order="1" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_60" name="Rate Law for R22" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (km0*M*T)+(kmM*Mact*T)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_373" name="km0" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_374" name="M" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_372" name="T" order="2" role="substrate"/>
        <ParameterDescription key="FunctionParameter_371" name="kmM" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_377" name="Mact" order="4" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_61" name="Rate Law for R23" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        rM+(ractM*g(X,Kx))
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_382" name="rM" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_381" name="ractM" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_380" name="X" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_379" name="Kx" order="3" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_62" name="g" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

      </MiriamAnnotation>
      <Expression>
        S/(S+K)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_385" name="S" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_384" name="K" order="1" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_63" name="Rate Law for R24" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        kM0*M*(V+(D/C1))
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_378" name="kM0" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_383" name="M" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_386" name="V" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_387" name="D" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_388" name="C1" order="4" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_64" name="Rate Law for R25" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (muM*M)+(kM0*M*T)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_393" name="muM" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_392" name="M" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_391" name="kM0" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_390" name="T" order="3" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_65" name="Rate Law for R26" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (pT*Mact)+(muMact*Mact)+Km*Mact*T
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_396" name="pT" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_395" name="Mact" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_394" name="muMact" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_389" name="Km" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_397" name="T" order="4" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_66" name="Rate Law for R27" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
      </MiriamAnnotation>
      <Expression>
        (1/C1)*((pVact*Iact)+(pV*I)-(((kM0*M)+(kM*Mact)*V)))
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_402" name="C1" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_401" name="pVact" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_400" name="Iact" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_399" name="pV" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_398" name="I" order="4" role="modifier"/>
        <ParameterDescription key="FunctionParameter_403" name="kM0" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_404" name="M" order="6" role="modifier"/>
        <ParameterDescription key="FunctionParameter_405" name="kM" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_406" name="Mact" order="8" role="modifier"/>
        <ParameterDescription key="FunctionParameter_407" name="V" order="9" role="product"/>
      </ListOfParameterDescriptions>
    </Function>
  </ListOfFunctions>
  <Model key="Model_1" name="Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium" simulationType="time" timeUnit="d" volumeUnit="l" areaUnit="m²" lengthUnit="m" quantityUnit="pmol" type="deterministic" avogadroConstant="6.0221407599999999e+23">
    <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:vCard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <rdf:Description rdf:about="#Model_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:efo:EFO:0007224"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C3439"/>
    <bqbiol:hasTaxon rdf:resource="urn:miriam:taxonomy:9606"/>
    <dcterms:bibliographicCitation>
      <rdf:Description>
        <CopasiMT:isDescribedBy rdf:resource="urn:miriam:pubmed:34430043"/>
      </rdf:Description>
    </dcterms:bibliographicCitation>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2021-10-29T14:48:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <dcterms:creator>
      <rdf:Description>
        <vCard:EMAIL>ktiwari@ebi.ac.uk</vCard:EMAIL>
        <vCard:N>
          <rdf:Description>
            <vCard:Family>Tiwari</vCard:Family>
            <vCard:Given>Krishna</vCard:Given>
          </rdf:Description>
        </vCard:N>
        <vCard:ORG>
          <rdf:Description>
            <vCard:Orgname>EMBL-EBI</vCard:Orgname>
          </rdf:Description>
        </vCard:ORG>
      </rdf:Description>
    </dcterms:creator>
    <CopasiMT:occursIn rdf:resource="urn:miriam:bto:BTO:0003511"/>
  </rdf:Description>
</rdf:RDF>

    </MiriamAnnotation>
    <ListOfCompartments>
      <Compartment key="Compartment_0" name="compartment" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Compartment_0">
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Compartment>
    </ListOfCompartments>
    <ListOfMetabolites>
      <Metabolite key="Metabolite_0" name="A1" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
        <Comment>
          type I alveolar cells
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_1" name="A2+" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
        <Comment>
          susceptible (ACE2-positive) type II alveolar cells
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_2" name="A2+act" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
        <Comment>
          ACE2-positive type II alveolar cells that are stimulated by interferons
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_3" name="I" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <Comment>
          infectious type II alveolar cells
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_4" name="A2" simulationType="assignment" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
        <Comment>
          total type II alveolar cells
        </Comment>
        <Expression>
          &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2+],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2+act],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2-],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2-act],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_5" name="A2-" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
        <Comment>
          immune (ACE2-negative) type II alveolar cells
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_6" name="A2-act" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
        <Comment>
          ACE2-negative type II alveolar cells that are stimulated by interferons
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_7" name="I_act" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <Comment>
          infectious type II alveolar cells stimulated by interferons
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_8" name="D" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <Comment>
          apoptotic alveolar cells
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_9" name="F" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <Comment>
          concentration of interferons
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_10" name="X" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <Comment>
          concentration of chemokines
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_11" name="T" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <Comment>
          concentration of toxins
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_12" name="M" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <Comment>
          inactivated innate immune cells
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_13" name="M_act" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <Comment>
          activated innate immune cells
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_14" name="V" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <Comment>
          concentration of free virus
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_15" name="A" simulationType="assignment" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
        <Comment>
          total concentration of alveolar cells (in pM) that are not yet treated by interferons
        </Comment>
        <Expression>
          (&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2+],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2-],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A1],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[I],Reference=Concentration>)/&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[C1],Reference=Value>*1e-2/6.02
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_16" name="AT" simulationType="assignment" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A1],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2+],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2-],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2+act],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2-act],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[I],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[I_act],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_17" name="A2act" simulationType="assignment" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2-act],Reference=Concentration>+&lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2+act],Reference=Concentration>
        </Expression>
      </Metabolite>
    </ListOfMetabolites>
    <ListOfModelValues>
      <ModelValue key="ModelValue_0" name="C1" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_1" name="C2" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_2" name="muA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_3" name="muM" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_4" name="mu+M" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_5" name="r2" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_6" name="gamma" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_7" name="theta" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_8" name="Ka1" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_9" name="p+" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_10" name="a2+" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_11" name="a2-" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_12" name="KA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_13" name="rM" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_14" name="rM+" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_15" name="beta" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_16" name="Kv" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_17" name="qv" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_18" name="muV" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_19" name="pV" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_20" name="pV+" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_21" name="u" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_22" name="mu1" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_23" name="kM0" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_24" name="kM" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_25" name="pX" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_26" name="Kx" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_27" name="pf1" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_28" name="pf2" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_29" name="muF" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_30" name="alpha" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_31" name="Kf" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_32" name="qF" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_33" name="pT" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_34" name="r" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_35" name="Kt" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_36" name="muX" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_37" name="muT" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_38" name="T0" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_39" name="q" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_39">
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
    </ListOfModelValues>
    <ListOfReactions>
      <Reaction key="Reaction_0" name="R1" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_0" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_4" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4970" name="alpha" value="0.6"/>
        </ListOfConstants>
        <KineticLaw function="Function_42" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_269">
              <SourceParameter reference="ModelValue_30"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_268">
              <SourceParameter reference="Metabolite_4"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_1" name="R2" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_0" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4969" name="r" value="0.1"/>
          <Constant key="Parameter_4968" name="muA" value="0.00035"/>
          <Constant key="Parameter_4967" name="KT" value="300"/>
        </ListOfConstants>
        <KineticLaw function="Function_43" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_270">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_250">
              <SourceParameter reference="Metabolite_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_271">
              <SourceParameter reference="ModelValue_34"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_272">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_273">
              <SourceParameter reference="ModelValue_35"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_2" name="R3" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_16" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_5" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4966" name="Ka" value="5.3e+10"/>
          <Constant key="Parameter_4965" name="a2" value="2.8e-05"/>
          <Constant key="Parameter_4964" name="mu" value="0.005"/>
          <Constant key="Parameter_4963" name="r2" value="0.055"/>
        </ListOfConstants>
        <KineticLaw function="Function_45" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_285">
              <SourceParameter reference="ModelValue_5"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_284">
              <SourceParameter reference="Metabolite_16"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_283">
              <SourceParameter reference="ModelValue_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_282">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_281">
              <SourceParameter reference="ModelValue_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_280">
              <SourceParameter reference="Metabolite_5"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_286">
              <SourceParameter reference="ModelValue_21"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_287">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_3" name="R10" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4962" name="muA" value="0.00035"/>
          <Constant key="Parameter_4961" name="u" value="0.005"/>
          <Constant key="Parameter_4960" name="r" value="0.1"/>
          <Constant key="Parameter_4959" name="a" value="0.6"/>
          <Constant key="Parameter_4958" name="KT" value="300"/>
        </ListOfConstants>
        <KineticLaw function="Function_51" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_327">
              <SourceParameter reference="ModelValue_21"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_326">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_325">
              <SourceParameter reference="ModelValue_30"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_324">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_323">
              <SourceParameter reference="ModelValue_34"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_309">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_328">
              <SourceParameter reference="ModelValue_35"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_4" name="R11" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_6" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4957" name="muA" value="0.00035"/>
          <Constant key="Parameter_4956" name="u" value="0.005"/>
          <Constant key="Parameter_4955" name="r" value="0.1"/>
          <Constant key="Parameter_4954" name="a" value="0.6"/>
          <Constant key="Parameter_4953" name="KT" value="300"/>
        </ListOfConstants>
        <KineticLaw function="Function_51" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_327">
              <SourceParameter reference="ModelValue_21"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_326">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_325">
              <SourceParameter reference="ModelValue_30"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_324">
              <SourceParameter reference="Metabolite_6"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_323">
              <SourceParameter reference="ModelValue_34"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_309">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_328">
              <SourceParameter reference="ModelValue_35"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_5" name="R12" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4952" name="mu1" value="0.013888"/>
          <Constant key="Parameter_4951" name="muA" value="0.00035"/>
          <Constant key="Parameter_4950" name="r" value="0.1"/>
          <Constant key="Parameter_4949" name="Kt" value="300"/>
        </ListOfConstants>
        <KineticLaw function="Function_52" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_335">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_334">
              <SourceParameter reference="ModelValue_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_333">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_332">
              <SourceParameter reference="ModelValue_34"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_331">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_330">
              <SourceParameter reference="ModelValue_35"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_6" name="R13" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_9" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4948" name="q" value="40"/>
          <Constant key="Parameter_4947" name="alpha" value="0.6"/>
          <Constant key="Parameter_4946" name="Kf" value="100"/>
        </ListOfConstants>
        <KineticLaw function="Function_53" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_340">
              <SourceParameter reference="ModelValue_30"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_339">
              <SourceParameter reference="Metabolite_9"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_338">
              <SourceParameter reference="ModelValue_32"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_337">
              <SourceParameter reference="Metabolite_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_336">
              <SourceParameter reference="ModelValue_31"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_329">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_7" name="R14" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4945" name="muA" value="0.00035"/>
          <Constant key="Parameter_4944" name="r" value="0.1"/>
          <Constant key="Parameter_4943" name="mu1" value="0.013888"/>
          <Constant key="Parameter_4942" name="Kt" value="300"/>
        </ListOfConstants>
        <KineticLaw function="Function_54" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_346">
              <SourceParameter reference="ModelValue_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_345">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_344">
              <SourceParameter reference="Metabolite_7"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_343">
              <SourceParameter reference="ModelValue_34"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_342">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_341">
              <SourceParameter reference="ModelValue_35"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_8" name="R15" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_3" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_7" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4941" name="muA" value="0.00035"/>
          <Constant key="Parameter_4940" name="r" value="0.1"/>
          <Constant key="Parameter_4939" name="mu1" value="0.013888"/>
          <Constant key="Parameter_4938" name="Kt" value="100"/>
        </ListOfConstants>
        <KineticLaw function="Function_55" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_352">
              <SourceParameter reference="ModelValue_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_351">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_350">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_349">
              <SourceParameter reference="Metabolite_7"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_348">
              <SourceParameter reference="ModelValue_34"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_347">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_353">
              <SourceParameter reference="ModelValue_31"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_9" name="R16" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_12" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4937" name="C1" value="20"/>
          <Constant key="Parameter_4936" name="Km" value="0.0003"/>
          <Constant key="Parameter_4935" name="km0" value="0.0001"/>
        </ListOfConstants>
        <KineticLaw function="Function_56" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_360">
              <SourceParameter reference="ModelValue_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_359">
              <SourceParameter reference="Metabolite_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_358">
              <SourceParameter reference="ModelValue_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_357">
              <SourceParameter reference="ModelValue_24"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_356">
              <SourceParameter reference="Metabolite_13"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_10" name="R17" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_9" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_3" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_7" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4934" name="pf2" value="0"/>
          <Constant key="Parameter_4933" name="C2" value="0.25"/>
          <Constant key="Parameter_4932" name="pf1" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_57" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_363">
              <SourceParameter reference="ModelValue_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_362">
              <SourceParameter reference="ModelValue_28"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_361">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_354">
              <SourceParameter reference="Metabolite_7"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_355">
              <SourceParameter reference="ModelValue_27"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_364">
              <SourceParameter reference="Metabolite_13"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_11" name="R4" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4931" name="KT" value="300"/>
          <Constant key="Parameter_4930" name="a" value="0.6"/>
          <Constant key="Parameter_4929" name="a2" value="0.0005"/>
          <Constant key="Parameter_4928" name="muA" value="0.00035"/>
          <Constant key="Parameter_4927" name="r" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_46" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_295">
              <SourceParameter reference="ModelValue_34"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_294">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_293">
              <SourceParameter reference="ModelValue_35"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_292">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_291">
              <SourceParameter reference="ModelValue_30"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_290">
              <SourceParameter reference="ModelValue_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_289">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_12" name="R5" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_9" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4926" name="q" value="40"/>
          <Constant key="Parameter_4925" name="alpha" value="0.6"/>
          <Constant key="Parameter_4924" name="Kf" value="100"/>
        </ListOfConstants>
        <KineticLaw function="Function_44" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_278">
              <SourceParameter reference="ModelValue_30"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_277">
              <SourceParameter reference="Metabolite_9"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_276">
              <SourceParameter reference="ModelValue_32"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_275">
              <SourceParameter reference="Metabolite_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_274">
              <SourceParameter reference="ModelValue_31"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_279">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_13" name="R6" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_14" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4923" name="beta" value="0.16666"/>
          <Constant key="Parameter_4922" name="qV" value="1"/>
          <Constant key="Parameter_4921" name="C1" value="20"/>
          <Constant key="Parameter_4920" name="Kv" value="1e+09"/>
        </ListOfConstants>
        <KineticLaw function="Function_47" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_301">
              <SourceParameter reference="ModelValue_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_300">
              <SourceParameter reference="Metabolite_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_299">
              <SourceParameter reference="ModelValue_17"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_298">
              <SourceParameter reference="ModelValue_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_297">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_296">
              <SourceParameter reference="ModelValue_16"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_14" name="R7" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_5" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_16" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_1" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_6" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4919" name="Ka" value="5.3e+10"/>
          <Constant key="Parameter_4918" name="a2pos" value="0.0005"/>
          <Constant key="Parameter_4917" name="mu" value="0.005"/>
          <Constant key="Parameter_4916" name="r2" value="0.055"/>
        </ListOfConstants>
        <KineticLaw function="Function_48" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_306">
              <SourceParameter reference="ModelValue_5"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_305">
              <SourceParameter reference="Metabolite_16"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_304">
              <SourceParameter reference="ModelValue_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_303">
              <SourceParameter reference="Metabolite_5"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_302">
              <SourceParameter reference="ModelValue_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_288">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_307">
              <SourceParameter reference="ModelValue_21"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_308">
              <SourceParameter reference="Metabolite_6"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_15" name="R8" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_5" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4915" name="a2neg" value="2.8e-05"/>
          <Constant key="Parameter_4914" name="muA" value="0.00035"/>
          <Constant key="Parameter_4913" name="r" value="0.1"/>
          <Constant key="Parameter_4912" name="a" value="0.6"/>
          <Constant key="Parameter_4911" name="Kt" value="300"/>
        </ListOfConstants>
        <KineticLaw function="Function_49" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_316">
              <SourceParameter reference="ModelValue_30"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_315">
              <SourceParameter reference="ModelValue_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_314">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_313">
              <SourceParameter reference="Metabolite_5"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_312">
              <SourceParameter reference="ModelValue_34"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_311">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_310">
              <SourceParameter reference="ModelValue_35"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_16" name="R9" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_5" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_6" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_9" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4910" name="q" value="0"/>
          <Constant key="Parameter_4909" name="alpha" value="0.6"/>
          <Constant key="Parameter_4908" name="Kf" value="100"/>
        </ListOfConstants>
        <KineticLaw function="Function_50" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_322">
              <SourceParameter reference="ModelValue_30"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_321">
              <SourceParameter reference="Metabolite_9"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_320">
              <SourceParameter reference="ModelValue_39"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_319">
              <SourceParameter reference="Metabolite_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_318">
              <SourceParameter reference="ModelValue_31"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_317">
              <SourceParameter reference="Metabolite_5"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_17" name="R18" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_9" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4907" name="k1" value="0.35"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_29"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_9"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_18" name="R19" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_3" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_7" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_17" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4906" name="C2" value="0.25"/>
          <Constant key="Parameter_4905" name="px" value="0.006"/>
        </ListOfConstants>
        <KineticLaw function="Function_58" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_370">
              <SourceParameter reference="ModelValue_25"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_369">
              <SourceParameter reference="ModelValue_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_368">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_367">
              <SourceParameter reference="Metabolite_7"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_366">
              <SourceParameter reference="Metabolite_17"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_365">
              <SourceParameter reference="Metabolite_13"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_19" name="R20" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4904" name="k1" value="1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_36"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_10"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_20" name="R21" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4903" name="pT" value="0.12"/>
        </ListOfConstants>
        <KineticLaw function="Function_59" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_376">
              <SourceParameter reference="ModelValue_33"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_375">
              <SourceParameter reference="Metabolite_13"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_21" name="R22" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_12" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4902" name="kmM" value="0.0003"/>
          <Constant key="Parameter_4901" name="km0" value="0.0001"/>
        </ListOfConstants>
        <KineticLaw function="Function_60" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_373">
              <SourceParameter reference="ModelValue_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_374">
              <SourceParameter reference="Metabolite_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_372">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_371">
              <SourceParameter reference="ModelValue_24"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_377">
              <SourceParameter reference="Metabolite_13"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_22" name="R23" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_12" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4900" name="ractM" value="3.5e+08"/>
          <Constant key="Parameter_4899" name="rM" value="3e+06"/>
          <Constant key="Parameter_4898" name="Kx" value="500"/>
        </ListOfConstants>
        <KineticLaw function="Function_61" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_382">
              <SourceParameter reference="ModelValue_13"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_381">
              <SourceParameter reference="ModelValue_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_380">
              <SourceParameter reference="Metabolite_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_379">
              <SourceParameter reference="ModelValue_26"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_23" name="R24" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_12" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_14" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4897" name="C1" value="20"/>
          <Constant key="Parameter_4896" name="kM0" value="0.0001"/>
        </ListOfConstants>
        <KineticLaw function="Function_63" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_378">
              <SourceParameter reference="ModelValue_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_383">
              <SourceParameter reference="Metabolite_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_386">
              <SourceParameter reference="Metabolite_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_387">
              <SourceParameter reference="Metabolite_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_388">
              <SourceParameter reference="ModelValue_0"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_24" name="R25" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_12" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4895" name="kM0" value="0.0001"/>
          <Constant key="Parameter_4894" name="muM" value="0.0005"/>
        </ListOfConstants>
        <KineticLaw function="Function_64" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_393">
              <SourceParameter reference="ModelValue_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_392">
              <SourceParameter reference="Metabolite_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_391">
              <SourceParameter reference="ModelValue_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_390">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_25" name="R26" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4893" name="muMact" value="0.02"/>
          <Constant key="Parameter_4892" name="pT" value="0.12"/>
          <Constant key="Parameter_4891" name="Km" value="0.0003"/>
        </ListOfConstants>
        <KineticLaw function="Function_65" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_396">
              <SourceParameter reference="ModelValue_33"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_395">
              <SourceParameter reference="Metabolite_13"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_394">
              <SourceParameter reference="ModelValue_4"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_389">
              <SourceParameter reference="ModelValue_24"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_397">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_26" name="R27" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_14" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_7" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_3" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_12" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4890" name="pVact" value="0.0318"/>
          <Constant key="Parameter_4889" name="C1" value="20"/>
          <Constant key="Parameter_4888" name="pV" value="3.18"/>
          <Constant key="Parameter_4887" name="kM0" value="0.0001"/>
          <Constant key="Parameter_4886" name="kM" value="0.0003"/>
        </ListOfConstants>
        <KineticLaw function="Function_66" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_402">
              <SourceParameter reference="ModelValue_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_401">
              <SourceParameter reference="ModelValue_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_400">
              <SourceParameter reference="Metabolite_7"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_399">
              <SourceParameter reference="ModelValue_19"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_398">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_403">
              <SourceParameter reference="ModelValue_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_404">
              <SourceParameter reference="Metabolite_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_405">
              <SourceParameter reference="ModelValue_24"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_406">
              <SourceParameter reference="Metabolite_13"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_407">
              <SourceParameter reference="Metabolite_14"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_27" name="R28" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_14" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4885" name="k1" value="0.3333"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_18"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_14"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
    </ListOfReactions>
    <ListOfModelParameterSets activeSet="ModelParameterSet_1">
      <ModelParameterSet key="ModelParameterSet_1" name="Initial State">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelParameterSet_1">
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ModelParameterGroup cn="String=Initial Time" type="Group">
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium" value="0" type="Model" simulationType="time"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Compartment Sizes" type="Group">
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment]" value="1" type="Compartment" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Species Values" type="Group">
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A1]" value="1.1803395889600003e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2+]" value="9.9064215502000003e+20" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2+act]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[I]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2]" value="1.9812843100399999e+22" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2-]" value="1.882220094538e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2-act]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[I_act]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[D]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[F]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[X]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[T]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[M]" value="3.6072623152399998e+21" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[M_act]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[V]" value="120442815200000" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A]" value="2.6259334709302328e+18" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[AT]" value="3.1616238989999998e+22" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2act]" value="0" type="Species" simulationType="assignment"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Global Quantities" type="Group">
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[C1]" value="20" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[C2]" value="0.25" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muA]" value="0.00035" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muM]" value="0.00050000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[mu+M]" value="0.02" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r2]" value="0.055" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[gamma]" value="7.7300000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[theta]" value="0.0060000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Ka1]" value="20320" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[p+]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[a2+]" value="0.00050000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[a2-]" value="2.8e-05" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[KA]" value="53000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[rM]" value="3" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[rM+]" value="350" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[beta]" value="0.16666" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kv]" value="1000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[qv]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muV]" value="0.33329999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pV]" value="3.1800000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pV+]" value="0.031800000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[u]" value="0.0050000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[mu1]" value="0.013887999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM0]" value="0.0001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM]" value="0.00029999999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pX]" value="0.0060000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kx]" value="500" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pf1]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pf2]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muF]" value="0.34999999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[alpha]" value="0.59999999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kf]" value="100" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[qF]" value="40" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pT]" value="0.12" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r]" value="0.10000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kt]" value="300" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muX]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muT]" value="0.28999999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[T0]" value="5" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[q]" value="0" type="ModelValue" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Kinetic Parameters" type="Group">
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R1]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R1],ParameterGroup=Parameters,Parameter=alpha" value="0.59999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R2]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R2],ParameterGroup=Parameters,Parameter=r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R2],ParameterGroup=Parameters,Parameter=muA" value="0.00035" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muA],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R2],ParameterGroup=Parameters,Parameter=KT" value="300" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kt],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R3]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R3],ParameterGroup=Parameters,Parameter=Ka" value="53000" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[KA],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R3],ParameterGroup=Parameters,Parameter=a2" value="2.8e-05" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[a2-],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R3],ParameterGroup=Parameters,Parameter=mu" value="0.0050000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[u],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R3],ParameterGroup=Parameters,Parameter=r2" value="0.055" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R10]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R10],ParameterGroup=Parameters,Parameter=muA" value="0.00035" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muA],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R10],ParameterGroup=Parameters,Parameter=u" value="0.0050000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[u],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R10],ParameterGroup=Parameters,Parameter=r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R10],ParameterGroup=Parameters,Parameter=a" value="0.59999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R10],ParameterGroup=Parameters,Parameter=KT" value="300" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kt],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R11]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R11],ParameterGroup=Parameters,Parameter=muA" value="0.00035" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muA],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R11],ParameterGroup=Parameters,Parameter=u" value="0.0050000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[u],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R11],ParameterGroup=Parameters,Parameter=r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R11],ParameterGroup=Parameters,Parameter=a" value="0.59999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R11],ParameterGroup=Parameters,Parameter=KT" value="300" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kt],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R12]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R12],ParameterGroup=Parameters,Parameter=mu1" value="0.013887999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[mu1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R12],ParameterGroup=Parameters,Parameter=muA" value="0.00035" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muA],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R12],ParameterGroup=Parameters,Parameter=r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R12],ParameterGroup=Parameters,Parameter=Kt" value="300" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kt],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R13]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R13],ParameterGroup=Parameters,Parameter=q" value="40" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[qF],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R13],ParameterGroup=Parameters,Parameter=alpha" value="0.59999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R13],ParameterGroup=Parameters,Parameter=Kf" value="100" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kf],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R14]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R14],ParameterGroup=Parameters,Parameter=muA" value="0.00035" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muA],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R14],ParameterGroup=Parameters,Parameter=r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R14],ParameterGroup=Parameters,Parameter=mu1" value="0.013887999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[mu1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R14],ParameterGroup=Parameters,Parameter=Kt" value="300" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kt],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R15]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R15],ParameterGroup=Parameters,Parameter=muA" value="0.00035" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muA],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R15],ParameterGroup=Parameters,Parameter=r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R15],ParameterGroup=Parameters,Parameter=mu1" value="0.013887999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[mu1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R15],ParameterGroup=Parameters,Parameter=Kt" value="100" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kf],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R16]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R16],ParameterGroup=Parameters,Parameter=C1" value="20" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[C1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R16],ParameterGroup=Parameters,Parameter=Km" value="0.00029999999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R16],ParameterGroup=Parameters,Parameter=km0" value="0.0001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM0],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R17]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R17],ParameterGroup=Parameters,Parameter=pf2" value="0" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pf2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R17],ParameterGroup=Parameters,Parameter=C2" value="0.25" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[C2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R17],ParameterGroup=Parameters,Parameter=pf1" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pf1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R4]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R4],ParameterGroup=Parameters,Parameter=KT" value="300" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kt],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R4],ParameterGroup=Parameters,Parameter=a" value="0.59999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R4],ParameterGroup=Parameters,Parameter=a2" value="0.00050000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[a2+],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R4],ParameterGroup=Parameters,Parameter=muA" value="0.00035" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muA],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R4],ParameterGroup=Parameters,Parameter=r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R5]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R5],ParameterGroup=Parameters,Parameter=q" value="40" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[qF],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R5],ParameterGroup=Parameters,Parameter=alpha" value="0.59999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R5],ParameterGroup=Parameters,Parameter=Kf" value="100" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kf],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R6]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R6],ParameterGroup=Parameters,Parameter=beta" value="0.16666" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[beta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R6],ParameterGroup=Parameters,Parameter=qV" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[qv],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R6],ParameterGroup=Parameters,Parameter=C1" value="20" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[C1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R6],ParameterGroup=Parameters,Parameter=Kv" value="1000" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kv],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R7]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R7],ParameterGroup=Parameters,Parameter=Ka" value="53000" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[KA],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R7],ParameterGroup=Parameters,Parameter=a2pos" value="0.00050000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[a2+],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R7],ParameterGroup=Parameters,Parameter=mu" value="0.0050000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[u],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R7],ParameterGroup=Parameters,Parameter=r2" value="0.055" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R8]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R8],ParameterGroup=Parameters,Parameter=a2neg" value="2.8e-05" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[a2-],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R8],ParameterGroup=Parameters,Parameter=muA" value="0.00035" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muA],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R8],ParameterGroup=Parameters,Parameter=r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R8],ParameterGroup=Parameters,Parameter=a" value="0.59999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R8],ParameterGroup=Parameters,Parameter=Kt" value="300" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kt],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R9]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R9],ParameterGroup=Parameters,Parameter=q" value="0" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[q],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R9],ParameterGroup=Parameters,Parameter=alpha" value="0.59999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R9],ParameterGroup=Parameters,Parameter=Kf" value="100" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kf],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R18]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R18],ParameterGroup=Parameters,Parameter=k1" value="0.34999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muF],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R19]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R19],ParameterGroup=Parameters,Parameter=C2" value="0.25" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[C2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R19],ParameterGroup=Parameters,Parameter=px" value="0.0060000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pX],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R20]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R20],ParameterGroup=Parameters,Parameter=k1" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muX],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R21]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R21],ParameterGroup=Parameters,Parameter=pT" value="0.12" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pT],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R22]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R22],ParameterGroup=Parameters,Parameter=kmM" value="0.00029999999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R22],ParameterGroup=Parameters,Parameter=km0" value="0.0001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM0],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R23]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R23],ParameterGroup=Parameters,Parameter=ractM" value="350" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[rM+],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R23],ParameterGroup=Parameters,Parameter=rM" value="3" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[rM],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R23],ParameterGroup=Parameters,Parameter=Kx" value="500" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[Kx],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R24]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R24],ParameterGroup=Parameters,Parameter=C1" value="20" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[C1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R24],ParameterGroup=Parameters,Parameter=kM0" value="0.0001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM0],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R25]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R25],ParameterGroup=Parameters,Parameter=kM0" value="0.0001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM0],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R25],ParameterGroup=Parameters,Parameter=muM" value="0.00050000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muM],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R26]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R26],ParameterGroup=Parameters,Parameter=muMact" value="0.02" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[mu+M],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R26],ParameterGroup=Parameters,Parameter=pT" value="0.12" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pT],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R26],ParameterGroup=Parameters,Parameter=Km" value="0.00029999999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R27]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R27],ParameterGroup=Parameters,Parameter=pVact" value="0.031800000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pV+],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R27],ParameterGroup=Parameters,Parameter=C1" value="20" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[C1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R27],ParameterGroup=Parameters,Parameter=pV" value="3.1800000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[pV],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R27],ParameterGroup=Parameters,Parameter=kM0" value="0.0001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM0],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R27],ParameterGroup=Parameters,Parameter=kM" value="0.00029999999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[kM],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R28]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Reactions[R28],ParameterGroup=Parameters,Parameter=k1" value="0.33329999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Values[muV],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
        </ModelParameterGroup>
      </ModelParameterSet>
    </ListOfModelParameterSets>
    <StateTemplate>
      <StateTemplateVariable objectReference="Model_1"/>
      <StateTemplateVariable objectReference="Metabolite_1"/>
      <StateTemplateVariable objectReference="Metabolite_5"/>
      <StateTemplateVariable objectReference="Metabolite_12"/>
      <StateTemplateVariable objectReference="Metabolite_3"/>
      <StateTemplateVariable objectReference="Metabolite_0"/>
      <StateTemplateVariable objectReference="Metabolite_8"/>
      <StateTemplateVariable objectReference="Metabolite_9"/>
      <StateTemplateVariable objectReference="Metabolite_10"/>
      <StateTemplateVariable objectReference="Metabolite_11"/>
      <StateTemplateVariable objectReference="Metabolite_14"/>
      <StateTemplateVariable objectReference="Metabolite_2"/>
      <StateTemplateVariable objectReference="Metabolite_13"/>
      <StateTemplateVariable objectReference="Metabolite_6"/>
      <StateTemplateVariable objectReference="Metabolite_7"/>
      <StateTemplateVariable objectReference="Metabolite_4"/>
      <StateTemplateVariable objectReference="Metabolite_15"/>
      <StateTemplateVariable objectReference="Metabolite_16"/>
      <StateTemplateVariable objectReference="Metabolite_17"/>
      <StateTemplateVariable objectReference="Compartment_0"/>
      <StateTemplateVariable objectReference="ModelValue_0"/>
      <StateTemplateVariable objectReference="ModelValue_1"/>
      <StateTemplateVariable objectReference="ModelValue_2"/>
      <StateTemplateVariable objectReference="ModelValue_3"/>
      <StateTemplateVariable objectReference="ModelValue_4"/>
      <StateTemplateVariable objectReference="ModelValue_5"/>
      <StateTemplateVariable objectReference="ModelValue_6"/>
      <StateTemplateVariable objectReference="ModelValue_7"/>
      <StateTemplateVariable objectReference="ModelValue_8"/>
      <StateTemplateVariable objectReference="ModelValue_9"/>
      <StateTemplateVariable objectReference="ModelValue_10"/>
      <StateTemplateVariable objectReference="ModelValue_11"/>
      <StateTemplateVariable objectReference="ModelValue_12"/>
      <StateTemplateVariable objectReference="ModelValue_13"/>
      <StateTemplateVariable objectReference="ModelValue_14"/>
      <StateTemplateVariable objectReference="ModelValue_15"/>
      <StateTemplateVariable objectReference="ModelValue_16"/>
      <StateTemplateVariable objectReference="ModelValue_17"/>
      <StateTemplateVariable objectReference="ModelValue_18"/>
      <StateTemplateVariable objectReference="ModelValue_19"/>
      <StateTemplateVariable objectReference="ModelValue_20"/>
      <StateTemplateVariable objectReference="ModelValue_21"/>
      <StateTemplateVariable objectReference="ModelValue_22"/>
      <StateTemplateVariable objectReference="ModelValue_23"/>
      <StateTemplateVariable objectReference="ModelValue_24"/>
      <StateTemplateVariable objectReference="ModelValue_25"/>
      <StateTemplateVariable objectReference="ModelValue_26"/>
      <StateTemplateVariable objectReference="ModelValue_27"/>
      <StateTemplateVariable objectReference="ModelValue_28"/>
      <StateTemplateVariable objectReference="ModelValue_29"/>
      <StateTemplateVariable objectReference="ModelValue_30"/>
      <StateTemplateVariable objectReference="ModelValue_31"/>
      <StateTemplateVariable objectReference="ModelValue_32"/>
      <StateTemplateVariable objectReference="ModelValue_33"/>
      <StateTemplateVariable objectReference="ModelValue_34"/>
      <StateTemplateVariable objectReference="ModelValue_35"/>
      <StateTemplateVariable objectReference="ModelValue_36"/>
      <StateTemplateVariable objectReference="ModelValue_37"/>
      <StateTemplateVariable objectReference="ModelValue_38"/>
      <StateTemplateVariable objectReference="ModelValue_39"/>
    </StateTemplate>
    <InitialState type="initialState">
      0 9.9064215502000003e+20 1.882220094538e+22 3.6072623152399998e+21 0 1.1803395889600003e+22 0 0 0 0 120442815200000 0 0 0 0 1.9812843100399999e+22 2.6259334709302328e+18 3.1616238989999998e+22 0 1 20 0.25 0.00035 0.00050000000000000001 0.02 0.055 7.7300000000000004 0.0060000000000000001 20320 0.050000000000000003 0.00050000000000000001 2.8e-05 53000 3 350 0.16666 1000 1 0.33329999999999999 3.1800000000000002 0.031800000000000002 0.0050000000000000001 0.013887999999999999 0.0001 0.00029999999999999997 0.0060000000000000001 500 0.01 0 0.34999999999999998 0.59999999999999998 100 40 0.12 0.10000000000000001 300 1 0.28999999999999998 5 0 
    </InitialState>
  </Model>
  <ListOfTasks>
    <Task key="Task_17" name="Steady-State" type="steadyState" scheduled="false" updateModel="false">
      <Report reference="Report_11" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="JacobianRequested" type="bool" value="1"/>
        <Parameter name="StabilityAnalysisRequested" type="bool" value="1"/>
      </Problem>
      <Method name="Enhanced Newton" type="EnhancedNewton">
        <Parameter name="Resolution" type="unsignedFloat" value="1.0000000000000001e-09"/>
        <Parameter name="Derivation Factor" type="unsignedFloat" value="0.001"/>
        <Parameter name="Use Newton" type="bool" value="1"/>
        <Parameter name="Use Integration" type="bool" value="1"/>
        <Parameter name="Use Back Integration" type="bool" value="0"/>
        <Parameter name="Accept Negative Concentrations" type="bool" value="0"/>
        <Parameter name="Iteration Limit" type="unsignedInteger" value="50"/>
        <Parameter name="Maximum duration for forward integration" type="unsignedFloat" value="1000000000"/>
        <Parameter name="Maximum duration for backward integration" type="unsignedFloat" value="1000000"/>
        <Parameter name="Target Criterion" type="string" value="Distance and Rate"/>
      </Method>
    </Task>
    <Task key="Task_18" name="Time-Course" type="timeCourse" scheduled="false" updateModel="false">
      <Report reference="Report_12" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="300"/>
        <Parameter name="StepSize" type="float" value="0.10000000000000001"/>
        <Parameter name="Duration" type="float" value="30"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
        <Parameter name="Use Values" type="bool" value="0"/>
        <Parameter name="Values" type="string" value=""/>
      </Problem>
      <Method name="Deterministic (LSODA)" type="Deterministic(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="100000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
    <Task key="Task_19" name="Scan" type="scan" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="Subtask" type="unsignedInteger" value="1"/>
        <ParameterGroup name="ScanItems">
        </ParameterGroup>
        <Parameter name="Output in subtask" type="bool" value="1"/>
        <Parameter name="Adjust initial conditions" type="bool" value="0"/>
        <Parameter name="Continue on Error" type="bool" value="0"/>
      </Problem>
      <Method name="Scan Framework" type="ScanFramework">
      </Method>
    </Task>
    <Task key="Task_20" name="Elementary Flux Modes" type="fluxMode" scheduled="false" updateModel="false">
      <Report reference="Report_13" target="" append="1" confirmOverwrite="1"/>
      <Problem>
      </Problem>
      <Method name="EFM Algorithm" type="EFMAlgorithm">
      </Method>
    </Task>
    <Task key="Task_21" name="Optimization" type="optimization" scheduled="false" updateModel="false">
      <Report reference="Report_14" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Subtask" type="cn" value="CN=Root,Vector=TaskList[Steady-State]"/>
        <ParameterText name="ObjectiveExpression" type="expression">
          
        </ParameterText>
        <Parameter name="Maximize" type="bool" value="0"/>
        <Parameter name="Randomize Start Values" type="bool" value="0"/>
        <Parameter name="Calculate Statistics" type="bool" value="1"/>
        <ParameterGroup name="OptimizationItemList">
        </ParameterGroup>
        <ParameterGroup name="OptimizationConstraintList">
        </ParameterGroup>
      </Problem>
      <Method name="Random Search" type="RandomSearch">
        <Parameter name="Log Verbosity" type="unsignedInteger" value="0"/>
        <Parameter name="Number of Iterations" type="unsignedInteger" value="100000"/>
        <Parameter name="Random Number Generator" type="unsignedInteger" value="1"/>
        <Parameter name="Seed" type="unsignedInteger" value="0"/>
      </Method>
    </Task>
    <Task key="Task_22" name="Parameter Estimation" type="parameterFitting" scheduled="false" updateModel="false">
      <Report reference="Report_15" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Maximize" type="bool" value="0"/>
        <Parameter name="Randomize Start Values" type="bool" value="0"/>
        <Parameter name="Calculate Statistics" type="bool" value="1"/>
        <ParameterGroup name="OptimizationItemList">
        </ParameterGroup>
        <ParameterGroup name="OptimizationConstraintList">
        </ParameterGroup>
        <Parameter name="Steady-State" type="cn" value="CN=Root,Vector=TaskList[Steady-State]"/>
        <Parameter name="Time-Course" type="cn" value="CN=Root,Vector=TaskList[Time-Course]"/>
        <Parameter name="Create Parameter Sets" type="bool" value="0"/>
        <Parameter name="Use Time Sens" type="bool" value="0"/>
        <Parameter name="Time-Sens" type="cn" value=""/>
        <ParameterGroup name="Experiment Set">
        </ParameterGroup>
        <ParameterGroup name="Validation Set">
          <Parameter name="Weight" type="unsignedFloat" value="1"/>
          <Parameter name="Threshold" type="unsignedInteger" value="5"/>
        </ParameterGroup>
      </Problem>
      <Method name="Evolutionary Programming" type="EvolutionaryProgram">
        <Parameter name="Log Verbosity" type="unsignedInteger" value="0"/>
        <Parameter name="Number of Generations" type="unsignedInteger" value="200"/>
        <Parameter name="Population Size" type="unsignedInteger" value="20"/>
        <Parameter name="Random Number Generator" type="unsignedInteger" value="1"/>
        <Parameter name="Seed" type="unsignedInteger" value="0"/>
        <Parameter name="Stop after # Stalled Generations" type="unsignedInteger" value="0"/>
      </Method>
    </Task>
    <Task key="Task_23" name="Metabolic Control Analysis" type="metabolicControlAnalysis" scheduled="false" updateModel="false">
      <Report reference="Report_16" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Steady-State" type="key" value="Task_17"/>
      </Problem>
      <Method name="MCA Method (Reder)" type="MCAMethod(Reder)">
        <Parameter name="Modulation Factor" type="unsignedFloat" value="1.0000000000000001e-09"/>
        <Parameter name="Use Reder" type="bool" value="1"/>
        <Parameter name="Use Smallbone" type="bool" value="1"/>
      </Method>
    </Task>
    <Task key="Task_24" name="Lyapunov Exponents" type="lyapunovExponents" scheduled="false" updateModel="false">
      <Report reference="Report_17" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="ExponentNumber" type="unsignedInteger" value="3"/>
        <Parameter name="DivergenceRequested" type="bool" value="1"/>
        <Parameter name="TransientTime" type="float" value="0"/>
      </Problem>
      <Method name="Wolf Method" type="WolfMethod">
        <Parameter name="Orthonormalization Interval" type="unsignedFloat" value="1"/>
        <Parameter name="Overall time" type="unsignedFloat" value="1000"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
      </Method>
    </Task>
    <Task key="Task_25" name="Time Scale Separation Analysis" type="timeScaleSeparationAnalysis" scheduled="false" updateModel="false">
      <Report reference="Report_18" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
      </Problem>
      <Method name="ILDM (LSODA,Deuflhard)" type="TimeScaleSeparation(ILDM,Deuflhard)">
        <Parameter name="Deuflhard Tolerance" type="unsignedFloat" value="0.0001"/>
      </Method>
    </Task>
    <Task key="Task_26" name="Sensitivities" type="sensitivities" scheduled="false" updateModel="false">
      <Report reference="Report_19" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="SubtaskType" type="unsignedInteger" value="1"/>
        <ParameterGroup name="TargetFunctions">
          <Parameter name="SingleObject" type="cn" value=""/>
          <Parameter name="ObjectListType" type="unsignedInteger" value="7"/>
        </ParameterGroup>
        <ParameterGroup name="ListOfVariables">
          <ParameterGroup name="Variables">
            <Parameter name="SingleObject" type="cn" value=""/>
            <Parameter name="ObjectListType" type="unsignedInteger" value="41"/>
          </ParameterGroup>
          <ParameterGroup name="Variables">
            <Parameter name="SingleObject" type="cn" value=""/>
            <Parameter name="ObjectListType" type="unsignedInteger" value="0"/>
          </ParameterGroup>
        </ParameterGroup>
      </Problem>
      <Method name="Sensitivities Method" type="SensitivitiesMethod">
        <Parameter name="Delta factor" type="unsignedFloat" value="0.001"/>
        <Parameter name="Delta minimum" type="unsignedFloat" value="9.9999999999999998e-13"/>
      </Method>
    </Task>
    <Task key="Task_27" name="Moieties" type="moieties" scheduled="false" updateModel="false">
      <Report reference="Report_20" target="" append="1" confirmOverwrite="1"/>
      <Problem>
      </Problem>
      <Method name="Householder Reduction" type="Householder">
      </Method>
    </Task>
    <Task key="Task_28" name="Cross Section" type="crosssection" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
        <Parameter name="Use Values" type="bool" value="0"/>
        <Parameter name="Values" type="string" value=""/>
        <Parameter name="LimitCrossings" type="bool" value="0"/>
        <Parameter name="NumCrossingsLimit" type="unsignedInteger" value="0"/>
        <Parameter name="LimitOutTime" type="bool" value="0"/>
        <Parameter name="LimitOutCrossings" type="bool" value="0"/>
        <Parameter name="PositiveDirection" type="bool" value="1"/>
        <Parameter name="NumOutCrossingsLimit" type="unsignedInteger" value="0"/>
        <Parameter name="LimitUntilConvergence" type="bool" value="0"/>
        <Parameter name="ConvergenceTolerance" type="float" value="9.9999999999999995e-07"/>
        <Parameter name="Threshold" type="float" value="0"/>
        <Parameter name="DelayOutputUntilConvergence" type="bool" value="0"/>
        <Parameter name="OutputConvergenceTolerance" type="float" value="9.9999999999999995e-07"/>
        <ParameterText name="TriggerExpression" type="expression">
          
        </ParameterText>
        <Parameter name="SingleVariable" type="cn" value=""/>
      </Problem>
      <Method name="Deterministic (LSODA)" type="Deterministic(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="100000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
    <Task key="Task_29" name="Linear Noise Approximation" type="linearNoiseApproximation" scheduled="false" updateModel="false">
      <Report reference="Report_21" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Steady-State" type="key" value="Task_17"/>
      </Problem>
      <Method name="Linear Noise Approximation" type="LinearNoiseApproximation">
      </Method>
    </Task>
    <Task key="Task_30" name="Time-Course Sensitivities" type="timeSensitivities" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
        <Parameter name="Use Values" type="bool" value="0"/>
        <Parameter name="Values" type="string" value=""/>
        <ParameterGroup name="ListOfParameters">
        </ParameterGroup>
        <ParameterGroup name="ListOfTargets">
        </ParameterGroup>
      </Problem>
      <Method name="LSODA Sensitivities" type="Sensitivities(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
  </ListOfTasks>
  <ListOfReports>
    <Report key="Report_11" name="Steady-State" taskType="steadyState" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Steady-State]"/>
      </Footer>
    </Report>
    <Report key="Report_12" name="Time-Course" taskType="timeCourse" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Time-Course],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Time-Course],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_13" name="Elementary Flux Modes" taskType="fluxMode" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Elementary Flux Modes],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_14" name="Optimization" taskType="optimization" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Object=Description"/>
        <Object cn="String=\[Function Evaluations\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Value\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Parameters\]"/>
      </Header>
      <Body>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Function Evaluations"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Best Value"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Best Parameters"/>
      </Body>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_15" name="Parameter Estimation" taskType="parameterFitting" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Object=Description"/>
        <Object cn="String=\[Function Evaluations\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Value\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Parameters\]"/>
      </Header>
      <Body>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Function Evaluations"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Best Value"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Best Parameters"/>
      </Body>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_16" name="Metabolic Control Analysis" taskType="metabolicControlAnalysis" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Metabolic Control Analysis],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Metabolic Control Analysis],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_17" name="Lyapunov Exponents" taskType="lyapunovExponents" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Lyapunov Exponents],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Lyapunov Exponents],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_18" name="Time Scale Separation Analysis" taskType="timeScaleSeparationAnalysis" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Time Scale Separation Analysis],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Time Scale Separation Analysis],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_19" name="Sensitivities" taskType="sensitivities" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Sensitivities],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Sensitivities],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_20" name="Moieties" taskType="moieties" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Moieties],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Moieties],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_21" name="Linear Noise Approximation" taskType="linearNoiseApproximation" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Linear Noise Approximation],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Linear Noise Approximation],Object=Result"/>
      </Footer>
    </Report>
  </ListOfReports>
  <ListOfPlots>
    <PlotSpecification name="Concentrations, Volumes, and Global Quantity Values" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[A1]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A1],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[A2+]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2+],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[A2+act]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2+act],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[I]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[I],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[A2]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[A2-]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2-],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[A2-act]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2-act],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[I_act]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[I_act],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[D]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[D],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[F]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[F],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[X]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[X],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[T]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[T],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[M]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[M],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[M_act]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[M_act],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[V]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[V],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[A]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[AT]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[AT],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[A2act]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium,Vector=Compartments[compartment],Vector=Metabolites[A2act],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
  </ListOfPlots>
  <GUI>
  </GUI>
  <ListOfUnitDefinitions>
    <UnitDefinition key="Unit_1" name="meter" symbol="m">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_0">
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        m
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_5" name="second" symbol="s">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_4">
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        s
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_13" name="Avogadro" symbol="Avogadro">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_12">
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        Avogadro
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_17" name="item" symbol="#">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_16">
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        #
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_35" name="liter" symbol="l">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_34">
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        0.001*m^3
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_41" name="mole" symbol="mol">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_40">
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        Avogadro*#
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_69" name="day" symbol="d">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_68">
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        86400*s
      </Expression>
    </UnitDefinition>
  </ListOfUnitDefinitions>
</COPASI>
